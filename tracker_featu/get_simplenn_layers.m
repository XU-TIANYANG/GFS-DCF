function feature_map = get_simplenn_layers(im, fparams, gparams)

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
        cnn_feat = vl_simplenn(fparams.net, gpuArray(im),[],[],'CuDNN',true, 'Mode', 'test');
    else
        cnn_feat = vl_simplenn(fparams.net, im, [], [], 'Mode', 'test');
    end
    
    feature_map = cell(1,1,length(fparams.output_layer));
    
    for k = 1:length(fparams.output_layer)
        if fparams.downsample_factor(k) == 1
            feature_map{k} = cnn_feat(fparams.output_layer(k) + 1).x(fparams.start_ind(k,1):fparams.end_ind(k,1), fparams.start_ind(k,2):fparams.end_ind(k,2), :, :);
        else
            feature_map{k} = vl_nnpool(cnn_feat(fparams.output_layer(k) + 1).x(fparams.start_ind(k,1):fparams.end_ind(k,1), fparams.start_ind(k,2):fparams.end_ind(k,2), :, :), fparams.downsample_factor(k), 'stride', fparams.downsample_factor(k), 'method', 'avg');
        end
    end
else
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
        im_aug = cat(4,im_aug,imrotate(im,15,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,-15,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,7,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,-7,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,22,'bilinear','crop'));
        im_aug =cat(4,im_aug,imrotate(im,-22,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,30,'bilinear','crop'));
        im_aug =cat(4,im_aug,imrotate(im,-30,'bilinear','crop'));
    end
    if fparams.augment.flip == 1
        im_aug = cat(4,im_aug,flipdim(im,2));
        im_aug = cat(4,im_aug,flipdim(imgaussfilt(im,2),2));
    end
    
    if gparams.use_gpu
        cnn_feat = vl_simplenn(fparams.net, gpuArray(im_aug),[],[],'CuDNN',true, 'Mode', 'test');
    else
        cnn_feat = vl_simplenn(fparams.net, im_aug, [], [], 'Mode', 'test');
    end
    
    feature_map = cell(1,1,length(fparams.output_layer));
    
    for k = 1:length(fparams.output_layer)
        if fparams.downsample_factor(k) == 1
            temp = cnn_feat(fparams.output_layer(k) + 1).x(fparams.start_ind(k,1):fparams.end_ind(k,1), fparams.start_ind(k,2):fparams.end_ind(k,2), :, :);
            feature_map{k} = mean(temp,4);
        else
            temp = vl_nnpool(cnn_feat(fparams.output_layer(k) + 1).x(fparams.start_ind(k,1):fparams.end_ind(k,1), fparams.start_ind(k,2):fparams.end_ind(k,2), :, :), fparams.downsample_factor(k), 'stride', fparams.downsample_factor(k), 'method', 'avg');
            feature_map{k} = mean(temp,4);
        end
    end
end

