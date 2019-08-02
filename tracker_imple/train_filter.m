%% Optimisation of Eqn.12
function [filter_model_f, spatial_selection, channel_selection] = train_filter(xlf, feature_info, yf, seq, params, filter_model_f)

if seq.frame == 1
    filter_model_f = cell(size(xlf));
    channel_selection = cell(size(xlf));
    spatial_selection = params.mask_window;
end

for k = 1: numel(xlf)
    model_xf = gather(xlf{k});
 
    % intialize the variables and parameters
    % <
    if (seq.frame == 1)
        filter_model_f{k} = zeros(size(model_xf));
        lambda3 = 1e-3;
        iter_max = 3;
        mu_max = 20;
    else
        lambda3 = params.lambda3(feature_info.feature_is_deep(k)+1);
        iter_max = 2;
        mu_max = 0.1;
    end
 
    filter_f = single(zeros(size(model_xf)));
    filter_prime_f = filter_f;
    gamma_f = filter_f;
    mu  = 1;
    % >
    
    % pre-compute the variables
    % <
    T = feature_info.data_sz(k)^2;
    S_xx = sum(conj(model_xf) .* model_xf, 3);
    Sfilter_pre_f = sum(conj(model_xf) .* filter_model_f{k}, 3);
    Sfx_pre_f = bsxfun(@times, model_xf, Sfilter_pre_f);
    % >
    
    iter = 1;
    while (iter <= iter_max)
        
        % subproblem Eqn.13.a
        % <
        D = S_xx + T * (mu + lambda3);
        Spx_f = sum(conj(model_xf) .* filter_prime_f, 3);
        Sgx_f = sum(conj(model_xf) .* gamma_f, 3);
        
        filter_f = ((1/(T*(mu + lambda3)) * bsxfun(@times,  yf{k}, model_xf)) - ((1/(mu + lambda3)) * gamma_f) +(mu/(mu + lambda3)) * filter_prime_f) + (lambda3/(mu + lambda3)) * filter_model_f{k} - ...
            bsxfun(@rdivide,(1/(T*(mu + lambda3)) * bsxfun(@times, model_xf, (S_xx .*  yf{k})) + (lambda3/(mu + lambda3)) * Sfx_pre_f - ...
            (1/(mu + lambda3))* (bsxfun(@times, model_xf, Sgx_f)) +(mu/(mu + lambda3))* (bsxfun(@times, model_xf, Spx_f))), D);
        % >
        
        if iter == iter_max && seq.frame > 1
            break;
        end
        
        % subproblem Eqn.13.b
        % pruning operators are employed to fix the selection ratio
        % <
        pmu = ifft2((mu * filter_f+ gamma_f), 'symmetric');
        
        if (seq.frame == 1)
            filter_prime = zeros(size(pmu));
            channel_selection{k} = ones(size(pmu));
            for i = 1:size(pmu,3)
                filter_prime(:,:,i) = spatial_selection{k} .* pmu(:,:,i);
            end
        else
            filter_prime = pmu;
            if feature_info.feature_is_deep(k)
                channel_selection{k} = max(0, mu-params.lambda1./(numel(pmu)*10*sqrt(sum(sum(filter_prime.^2,1),2))));
                [~,b] = sort(channel_selection{k}(:),'descend');
                if numel(b)>1
                    channel_selection{k}(b(ceil(params.channel_selection_rate(feature_info.feature_is_deep(k)+1)*numel(b)):end)) = 0;
                end
                filter_prime = repmat(channel_selection{k},size(filter_prime,1),size(filter_prime,2),1) .* filter_prime;
                
                spatial_selection{k} = max(0,mu - params.lambda2./(numel(pmu)*sqrt(sum(filter_prime.^2,3))));
                [~,b] = sort(spatial_selection{k}(:),'descend');
                spatial_selection{k}(b(ceil(params.spatial_selection_rate(feature_info.feature_is_deep(k)+1)*numel(b)+0.5):end)) = 0;
                filter_prime = repmat(spatial_selection{k},1,1,size(filter_prime,3)) .* filter_prime;
            else
                spatial_selection{k} = max(0,mu - params.lambda2./(numel(pmu)*sqrt(sum(filter_prime.^2,3))));
                [~,b] = sort(spatial_selection{k}(:),'descend');
                spatial_selection{k}(b(ceil(params.spatial_selection_rate(feature_info.feature_is_deep(k)+1)*numel(b)):end)) = 0;
                filter_prime = repmat(spatial_selection{k},1,1,size(filter_prime,3)) .* filter_prime;
                
                channel_selection{k} = max(0,mu-params.lambda1./(numel(pmu)*2*sqrt(sum(sum(filter_prime.^2,1),2))));
                [~,b] = sort(channel_selection{k}(:),'descend');
                if numel(b)>1
                    channel_selection{k}(b(ceil(params.channel_selection_rate(feature_info.feature_is_deep(k)+1)*numel(b)+0.5):end)) = 0;
                end
                filter_prime = repmat(channel_selection{k},size(filter_prime,1),size(filter_prime,2),1) .* filter_prime;
            end
        end
        filter_prime_f = fft2(filter_prime);
        % >
        
        % subproblem Eqn.13.c
        % <
        gamma_f = gamma_f + (mu * (filter_f - filter_prime_f));
        % >
        
        % update the penalty mu
        % <
        mu = min(1.5 * mu, mu_max);
        % >
        
        iter = iter+1;
    end
    
    % save the trained filters (robustness test)
    %filter_refilter_f{k} = filter_f+randn(size(filter_f))*mean(filter_f(:))*params.stability_factor(feature_info.feature_is_deep(k)+1);
    
    % update the filters Eqn.2
    % <
    if seq.frame == 1
        filter_model_f{k} = filter_f;
    else
        filter_model_f{k} = params.learning_rate(feature_info.feature_is_deep(k)+1)*filter_f ...
            + (1-params.learning_rate(feature_info.feature_is_deep(k)+1))*filter_model_f{k};
    end
    % >
end

end






