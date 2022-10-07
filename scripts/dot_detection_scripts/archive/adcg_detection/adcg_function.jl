using Plots
using SeqFISH_ADCG
using Images
using FileIO
using CSV
using DataFrames


#tile will be provided from adcg_batch.py
function adcg(img, output_path, channel)
    #parameters
    if channel == 1
        sigma_lb = 1.2
        sigma_ub = 1.6
        noise_mean = 0.0
        min_weight = 400.0
        tau = 10.0^5#2.0*10^12
        final_loss_improvement = 40.0#10.0
        max_iters = 200#10000
        max_cd_iters = 10
        
    elseif channel == 2
        sigma_lb = 1.4
        sigma_ub = 2.2
        noise_mean = 0.0
        min_weight = 400.0
        tau = 10.0^5#2.0*10^12
        final_loss_improvement = 40.0#10.0
        max_iters = 200#10000
        max_cd_iters = 10
        
    elseif channel == 3
        sigma_lb = 1.8
        sigma_ub = 2.4
        noise_mean = 0.0
        min_weight = 300.0
        tau = 10.0^5#2.0*10^12
        final_loss_improvement = 40.0#10.0
        max_iters = 200#10000
        max_cd_iters = 10
        
    elseif channel == 4
        sigma_lb = 1.8
        sigma_ub = 2.2
        noise_mean = 0.0
        min_weight = 300.0
        tau = 10.0^5#2.0*10^12
        final_loss_improvement = 40.0#10.0
        max_iters = 200#10000
        max_cd_iters = 10
    end
    #adcg on entire image
    df = fit_2048x2048_img_tiles(
        img,sigma_lb, sigma_ub, tau,final_loss_improvement, 
        min_weight, max_iters, max_cd_iters, noise_mean
    )
    
    #remove duplicates
    df_rem_length = 0
    while length(df[!,"x"]) != df_rem_length
        df_rem = remove_duplicates(df, img, sigma_lb, sigma_ub, tau, noise_mean, 2.0)
        df_rem_2 = remove_duplicates(df_rem, img, sigma_lb, sigma_ub, tau, noise_mean, 2.0)
        df = copy(df_rem)
        df_rem_length = length(df_rem_2[!,"x"])
    end
     
    #write df
    CSV.write(output_path,df)
end



