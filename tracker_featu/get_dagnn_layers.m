function feature_map = get_dagnn_layers(im, fparams, gparams)

% Get layers from a cnn.

if size(im,3) == 1
    im = repmat(im, [1 1 3]);
end

im_sample_size = size(im);

if gparams.augment == 0
    
    %preprocess the image
    if ~isequal(im_sample_size(1:2), fparams.net.meta.normalization.imageSize(1:2))
        im = imresize(single(im), fparams.net.meta.normalization.imageSize(1:2));
    else
        im = single(im);
    end
    
    % Normalize with average image
    im = bsxfun(@minus, im, fparams.net.meta.normalization.averageImage);
    
    if gparams.use_gpu
        im = gpuArray(im);
        fparams.net.eval({'data', im});
    else
        fparams.net.eval({'data', im});
    end
    
    feature_map = cell(1,1,length(fparams.output_var));
    
    for k = 1:length(fparams.output_var)
        if fparams.downsample_factor(k) == 1
            feature_map{k} = fparams.net.vars(fparams.output_var(k)).value(fparams.start_ind(1,1)...
                :fparams.end_ind(1,1), fparams.start_ind(1,2):fparams.end_ind(1,2), :, :);
        else
            feature_map{k} = vl_nnpool(fparams.net.vars(fparams.output_var(k)).value(fparams.start_ind(1,1)...
                :fparams.end_ind(1,1), fparams.start_ind(1,2):fparams.end_ind(1,2), :, :),...
                fparams.downsample_factor(k), 'stride', fparams.downsample_factor(k), 'method', 'avg');
        end
    end
else
    
     %preprocess the image
    if ~isequal(im_sample_size(1:2), fparams.net.meta.normalization.imageSize(1:2))
        im = imresize(single(im), fparams.net.meta.normalization.imageSize(1:2));
    else
        im = single(im);
    end
    
    % Normalize with average image
    im = bsxfun(@minus, im, fparams.net.meta.normalization.averageImage);
    
    im_aug = im;
    if fparams.augment.blur == 1
        im_aug = cat(4,im_aug,imgaussfilt(im,1));
        im_aug = cat(4,im_aug,imgaussfilt(im,2));
    end
    if fparams.augment.rotation == 1
        im_aug = cat(4,im_aug,imrotate(im,16,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,-16,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,8,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,-8,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,24,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,-24,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,32,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,-32,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,40,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,-40,'bilinear','crop'));
    end
    if fparams.augment.flip == 1
        im_aug = cat(4,im_aug,flipdim(im,2));
        im_aug = cat(4,im_aug,flipdim(imgaussfilt(im,2),2));
    end

    if gparams.use_gpu
        fparams.net.eval({'data', gpuArray(im_aug)});
    else
        fparams.net.eval({'data', im_aug});
    end
    
    
    feature_map = cell(1,1,length(fparams.output_var));
    
    for k = 1:length(fparams.output_var)
        if fparams.downsample_factor(k) == 1
            feature_map{k} = mean(fparams.net.vars(fparams.output_var(k)).value(fparams.start_ind(1,1)...
                :fparams.end_ind(1,1), fparams.start_ind(1,2):fparams.end_ind(1,2), :, :),4);
            
        else
            feature_map{k} = vl_nnpool(fparams.net.vars(fparams.output_var(k)).value(fparams.start_ind(1,1)...
                :fparams.end_ind(1,1), fparams.start_ind(1,2):fparams.end_ind(1,2), :, :),...
                fparams.downsample_factor(k), 'stride', fparams.downsample_factor(k), 'method', 'avg');
             feature_map{k} = mean( feature_map{k},4);
        end
    end
    
    
    
end
