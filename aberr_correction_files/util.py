import tifffile as tif
import numpy as np
import pandas as pd
import json
import re
import os
import string
from pathlib import Path, PurePath
import jmespath
from PIL import Image, ImageSequence, UnidentifiedImageError


def pil_imopen(fname, metadata=False):
    im = Image.open(fname)

    if metadata:
        return im, pil_getmetadata(im)
    else:
        return im


def pil_imread(
    fname,
    metadata=False,
    swapaxes=False,
    ensure_4d=True,
    backup=tif.imread,
    **kwargs
):
    md = None

    import warnings
    warnings.simplefilter('ignore', UserWarning)

    try:
        im = pil_imopen(fname)
        md = pil_getmetadata(im)
        imarr = pil_frames_to_ndarray(im)
    except (ValueError, UnidentifiedImageError) as e:
        if callable(backup):
            imarr = backup(fname, **kwargs)
        else:
            raise e

    if ensure_4d and imarr.ndim == 3:
        # assumes 1 Z
        imarr = imarr[:, None, :]

    if swapaxes and imarr.ndim == 4:
        imarr = imarr.swapaxes(0, 1)

    if metadata and md:
        return imarr, md
    else:
        return imarr


def pil_getmetadata(im, relevant_keys=None):
    """
    pil_getmetadata
    ---------------
    Given a PIL image sequence im, retrieve the metadata associated
    with each frame in the sequence. Only keep metadata keys specified
    in `relevant_keys` - which will default to ones that we need such as
    channel, slice information. There are many metadata keys which are
    useless / do not change frame to frame.
    Returns: List of dicts in order of frame index.
    """

    if str(relevant_keys).lower() == 'all':
        relevant_keys = None

    elif not isinstance(relevant_keys, list):

        relevant_keys = [
            'Andor sCMOS Camera-Exposure',  # Exposure time (ms)
            'Channel',                      # Channel name (wavelength)
            'ChannelIndex',                 # Channel index (number)
            'Frame',                        # Time slice (usually not used)
            'FrameIndex',                   # Time slice index (usually not used)
            'PixelSizeUm',                  # XY pixel size in microns
            'Position',                     # Position 
            'PositionIndex',                # Position index (MMStack_PosX)
            'PositionName',                 # Position name
            'Slice',                        # Z slice
            'SliceIndex'                    # Z slice index (same as Slice)
        ]

    frame_metadata = []

    for frame in ImageSequence.Iterator(im):

        # The JSON string is stored in a key named "unknown",
        # probably because it doesn't correspond to a standard
        # TIF tag number.
        if 'unknown' in frame.tag.named().keys():
            jsstr = frame.tag.named()['unknown'][0]
            jsdict = json.loads(jsstr)

            if relevant_keys:
                # Only keep the relevant keys
                rel_dict = {
                    k: jsdict.get(k)
                    for k in relevant_keys
                }
            else:
                rel_dict = jsdict

            frame_metadata.append(rel_dict)

    return frame_metadata


def pil2numpy(im, dtype=np.uint16):

    return np.frombuffer(im.tobytes(), dtype=dtype).reshape(im.size)


def pil_frames_to_ndarray(im, dtype=np.uint16):
    """
    pil_frames_to_ndarray
    -----------------
    Given a PIL image sequence, return a Numpy array that is correctly
    ordered and shaped as (n_channels, n_slices, ...) so that we can 
    process it in a consistent way.
    To do this, we look at the ChannelIndex and SliceIndex of each frame
    in the stack, and insert them one by one into the correct position
    of a 4D numpy array.
    """
    metadata = pil_getmetadata(im)

    if not metadata:
        raise ValueError('Supplied image lacks metadata used for '
            'forming the correct image shape. Was the image not '
            'taken from ImageJ/MicroManager?')

    # Gives a list of ChannelIndex for each frame
    cinds = jmespath.search('[].ChannelIndex', metadata)
    # Gives a list of SliceIndex for each frame
    zinds = jmespath.search('[].SliceIndex', metadata)

    if (len(cinds) != len(zinds)
        or any([c is None for c in cinds])
        or any([z is None for z in zinds])
    ):
        raise ValueError('SuppliedImage lacks `ChannelIndex` or '
                         '`SliceIndex` metadata required to form '
                         'properly shaped numpy array. Was the image not '
                         'taken directly from ImageJ/MicroManager?')

    ncs = max(cinds) + 1
    nzs = max(zinds) + 1

    total_frames = ncs * nzs
    assert total_frames == im.n_frames, 'wrong shape'

    # Concatenate the channel and slice count to the XY shape in im.size
    new_shape = (ncs, nzs) + im.size

    # Make an empty ndarray of the proper shape and dtype
    npoutput = np.empty(new_shape, dtype=dtype)

    # Loop in a nested fashion over channel first then Z slice
    for c in range(ncs):
        for z in range(nzs):

            # Find the frame whose ChannelIndex and SliceIndex
            # match the current c and z values
            entry = jmespath.search(
                f'[?ChannelIndex==`{c}` && SliceIndex==`{z}`]', metadata)[0]

            # Find the *index* of the matching frame so that we can insert it
            ind = metadata.index(entry)

            # Select the matching frame
            im.seek(ind)

            # Copy the frame into the correct c and z position in the numpy array
            npoutput[c, z] = pil2numpy(im)

    return npoutput


def safe_imread(fname, is_ome=False, is_imagej=True):
    imarr = np.array([])

    try:
        imarr = tif.imread(fname, is_ome=is_ome, is_imagej=is_imagej)
    except (AttributeError, RuntimeError):
        imarr = tif.imread(fname, is_ome=is_ome, is_imagej=False)

    return imarr


def safe_imwrite(
    arr,
    fname,
    compression='DEFLATE',
    ome=False,
    imagej=True
):
    arr = arr.copy()
    try:
        with tif.TiffWriter(fname, ome=ome, imagej=imagej) as tw:
            tw.write(arr, compression=compression)
    except RuntimeError:
        with tif.TiffWriter(fname, ome=ome, imagej=False) as tw:
            tw.write(arr, compression=compression)

    del arr


def populate_files(
        directory,
        dirs_only=True,
        prefix='MMStack_Pos',
        postfix='',
        converter=int
):
    """
    populate_files
    ------------------
    Takes either a *list* of files/folders OR a directory name
    and searches in it for entries that match `regex` of the form
    <Prefix><Number><Postfix>,capturing the number.
    Also takes `converter`, a function to convert the number from a string
    to a number. default is int(). If this fails it is kept as a string.
    Returns: List of tuples of the form (name, number), sorted by
    number.
    """
    regex = re.escape(prefix) + '(\d+)' + re.escape(postfix)
    pos_re = re.compile(regex)

    result = []

    def extract_match(name, regex=pos_re, converter=converter):
        m = regex.search(name)
        if m is not None:
            try:
                ret = m.group(0), converter(m.group(1))
            except ValueError:
                ret = m.group(0), m.group(1)

            return ret
        else:
            return None

    if isinstance(directory, list):
        dirs = directory
    else:
        if dirs_only:
            dirs = [entry.name for entry in os.scandir(directory)
                    if entry.is_dir()]
        else:
            dirs = [entry.name for entry in os.scandir(directory)]

    for d in dirs:
        m = extract_match(d)
        if m is not None:
            result.append(m)

    # sort by the number
    return sorted(result, key=lambda n: n[1])


def fmt2regex(fmt, delim=os.path.sep):
    """
    fmt2regex:
    convert a curly-brace format string with named fields
    into a regex that captures those fields as named groups,
    Returns:
    * reg: compiled regular expression to capture format fields as named groups
    * globstr: equivalent glob string (with * wildcards for each field) that can
        be used to find potential files that will be analyzed with reg.
    """
    sf = string.Formatter()

    regex = []
    globstr = []
    keys = set()

    numkey = 0

    fmt = str(fmt).rstrip(delim)

    if delim:
        parts = fmt.split(delim)
    else:
        delim = ''
        parts = [fmt]

    re_delim = re.escape(delim)

    for part in parts:
        part_regex = ''
        part_glob = ''

        for a in sf.parse(part):
            r = re.escape(a[0])

            newglob = a[0]
            if a[1]:
                newglob = newglob + '*'
            part_glob += newglob

            if a[1] is not None:
                k = re.escape(a[1])

                if len(k) == 0:
                    k = f'k{numkey}'
                    numkey += 1

                if k in keys:
                    r = r + f'(?P={k})'
                else:
                    r = r + f'(?P<{k}>[^{re_delim}]+)'

                keys.add(k)

            part_regex += r

        globstr.append(part_glob)
        regex.append(part_regex)

    reg = re.compile('^'+re_delim.join(regex))
    globstr = delim.join(globstr)

    return reg, globstr


def find_matching_files(base, fmt, paths=None):
    """
    findAllMatchingFiles: Starting within a base directory,
    find all files that match format `fmt` with named fields.
    Returns:
    * files: list of filenames, including `base`, that match fmt
    * keys: Dict of lists, where the keys are each named key from fmt,
        and the lists contain the value for each field of each file in `files`,
        in the same order as `files`.
    """

    reg, globstr = fmt2regex(fmt)

    base = PurePath(base)

    files = []
    mtimes = []
    keys = {}

    if paths is None:
        paths = Path(base).glob(globstr)
    else:
        paths = [Path(p) for p in paths]

    for f in paths:
        m = reg.match(str(f.relative_to(base)))

        if m:
            try:
                mtimes.append(os.stat(f).st_mtime)
            except (PermissionError, OSError):
                mtimes.append(-1)

            files.append(f)

            for k, v in m.groupdict().items():
                if k not in keys.keys():
                    keys[k] = []

                keys[k].append(v)

    return files, keys, mtimes


def fmts2file(*fmts, fields={}):
    fullpath = str(Path(*fmts))
    return Path(fullpath.format(**fields))


def k2f(
    k,
    delimiter='/'
):
    return PurePath(str(k).replace(delimiter, os.sep))


def f2k(
    f,
    delimiter='/'
):
    return str(f).replace(os.sep, delimiter)


def sanitize(
    k,
    delimiter='/',
    delimiter_allowed=True,
    raiseonfailure=False
):
    badchars = '\\[]{}^%#` <>~|'

    if raiseonfailure:
        err = delimiter in badchars or (delimiter in k and not delimiter_allowed)
        if err:
            raise ValueError(f'Delimiter:  {delimiter}  is not allowed.')

    if delimiter and delimiter_allowed:
        parts = k.split(delimiter)
    else:
        parts = [k]

    # put it in the middle so there's less chance of it messing up
    # the exp
    # Note: I added unix shell special characters like $, &, ;, *, ! because
    # the analysis names will be used as folder names
    exp = '[\\\\{^}%$&*@!/?;` ' + re.escape(delimiter) + '\\[\\]>~<#|]'
    parts_sanitized = [re.sub(exp, '', part) for part in parts]

    if any([len(part) == 0 for part in parts_sanitized]):
        raise ValueError(f'After sanitizing, a part of the string "{k}"'
                         f' disappeared completely.')

    return delimiter.join(parts_sanitized)


def ls_recursive(root='.', level=1, ignore=[], dirsonly=True, flat=False):
    if flat:
        result = []
    else:
        result = {}

    if not isinstance(level, int):
        raise ValueError('level must be an integer')

    def _ls_recursive(
        contents=None,
        folder='.',
        root='',
        maxlevel=level,
        curlevel=0,
        dirsonly=dirsonly,
        flat=flat
    ):
        if curlevel == maxlevel:
            if flat:
                contents.extend([f.relative_to(root) for f in Path(folder).iterdir()
                        if f.is_dir() or not dirsonly])
                return contents
            else:
                return [f.name for f in Path(folder).iterdir()
                        if f.is_dir() or not dirsonly]

        args = dict(
            contents=contents,
            root=root,
            maxlevel=level,
            curlevel=curlevel+1,
            dirsonly=dirsonly,
            flat=flat
        )

        subfolders =[f for f in Path(folder).iterdir() if (
            f.is_dir() and not any([f.match(p) for p in ignore]))]

        if flat:
            [_ls_recursive(folder=f, **args) for f in subfolders]
        else:
            contents = {f.name: _ls_recursive(folder=f, **args) for f in subfolders}

        return contents

    result = _ls_recursive(
        result,
        folder=root,
        root=root,
        maxlevel=level,
        curlevel=0,
        dirsonly=dirsonly,
        flat=flat
    )
    return result


def process_requires(requires):
    reqs = []

    for entry in requires:
        reqs.extend([r.strip() for r in entry.split('|')])

    return reqs


def source_keys_conv(sks):
    # convert a string rep of a list to an actual list
    return sks.split('|')


def process_file_entries(entries):
    result = {}

    for key, value in entries.items():
        info = dict.fromkeys([
            'pattern',
            'requires',
            'generator',
            'preupload'], None)

        if isinstance(value, str):
            info['pattern'] = value
        elif isinstance(value, dict):
            info.update(value)
        else:
            raise TypeError('Each file in config must be either a string or a dict')

        result[key] = info

    return result


def process_file_locations(locs):
    result = {}

    for key, value in locs.items():
        info = value

        dformat = info.get('dataset_format', '')

        dfr, dfg = fmt2regex(dformat)
        fields = list(dfr.groupindex.keys())

        info['dataset_format_re'] = dfr
        info['dataset_format_glob'] = dfg
        info['dataset_format_fields'] = fields
        info['dataset_format_nest'] = len(Path(dformat).parts) - 1

        result[key] = info

    return result


def empty_or_false(thing):
    if isinstance(thing, pd.DataFrame):
        return thing.empty

    return not thing


def notempty(dfs):
    return [not empty_or_false(df) for df in dfs]


def copy_or_nop(df):
    try:
        result = df.copy()
    except AttributeError:
        result = df

    return result


def sort_as_num_or_str(coll, numtype=int):
    np_coll = np.array(coll)

    try:
        result = np.sort(np_coll.astype(numtype)).astype(str)
    except ValueError:
        result = np.sort(np_coll.astype(str))

    return result