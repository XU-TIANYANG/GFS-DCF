function [features, gparams, feature_info] = init_features(features, gparams, params, is_color_image, img_sample_sz, size_mode)

if nargin < 3
    size_mode = 'same';
end

% Set missing global parameters to default values
if ~isfield(gparams, 'normalize_power')
    gparams.normalize_power = [];
end
if ~isfield(gparams, 'normalize_size')
    gparams.normalize_size = true;
end
if ~isfield(gparams, 'normalize_dim')
    gparams.normalize_dim = false;
end
if ~isfield(gparams, 'square_root_normalization')
    gparams.square_root_normalization = false;
end
if ~isfield(gparams, 'use_gpu')
    gparams.use_gpu = false;
end

% find which features to keep
feat_ind = false(length(features),1);
for n = 1:length(features)
    
    if ~isfield(features{n}.fparams,'useForColor')
        features{n}.fparams.useForColor = true;
    end
    
    if ~isfield(features{n}.fparams,'useForGray')
        features{n}.fparams.useForGray = true;
    end
    
    if (features{n}.fparams.useForColor && is_color_image) || (features{n}.fparams.useForGray && ~is_color_image)
        % keep feature
        feat_ind(n) = true;
    end
end

% remove features that are not used
features = features(feat_ind);

num_features = length(features);

feature_info.min_cell_size = zeros(num_features,1);

for k = 1:length(features)
    if isequal(features{k}.getFeature, @get_fhog)
        if ~isfield(features{k}.fparams, 'nOrients')
            features{k}.fparams.nOrients = 9;
        end
        features{k}.fparams.nDim = 3*features{k}.fparams.nOrients+5-1;
        features{k}.is_cell = false;
        features{k}.is_cnn = false;
        
    elseif isequal(features{k}.getFeature, @get_table_feature)
        table = load([features{k}.fparams.tablename]);
        features{k}.fparams.nDim = size(table.(features{k}.fparams.tablename),2);
        features{k}.is_cell = false;
        features{k}.is_cnn = false;
        
    elseif isequal(features{k}.getFeature, @get_simplenn_layers) 
        if ~isfield(features{k}.fparams, 'input_size_mode')
            features{k}.fparams.input_size_mode = 'adaptive';
        end
        if ~isfield(features{k}.fparams, 'input_size_scale')
            features{k}.fparams.input_size_scale = 1;
        end
        if ~isfield(features{k}.fparams, 'downsample_factor')
            features{k}.fparams.downsample_factor = ones(1, length(features{k}.fparams.output_layer));
        end
        
        net = load(features{k}.fparams.nn_name);
        net = vl_simplenn_tidy(net);
        net.layers = net.layers(1:max(features{k}.fparams.output_layer));
        base_input_sz = img_sample_sz(2,:);
        net.meta.normalization.imageSize(1:2) = round(base_input_sz);   
        net.meta.normalization.averageImageOrig = net.meta.normalization.averageImage;
        if isfield(net.meta,'inputSize')
            net.meta.inputSize = base_input_sz;
        end    
        if size(net.meta.normalization.averageImage,1) > 1 || size(net.meta.normalization.averageImage,2) > 1
            net.meta.normalization.averageImage = imresize(single(net.meta.normalization.averageImage), net.meta.normalization.imageSize(1:2));
        end
        net.info = vl_simplenn_display(net);    
        features{k}.fparams.nDim = net.info.dataSize(3, features{k}.fparams.output_layer+1)';    
        if isfield(net.info, 'receptiveFieldStride')
            net_info_stride = cat(2, [1; 1], net.info.receptiveFieldStride);
        else
            net_info_stride = [1; 1];
        end            
        features{k}.fparams.cell_size = net_info_stride(1, features{k}.fparams.output_layer+1)' .* features{k}.fparams.downsample_factor';    
        features{k}.is_cell = true;
        features{k}.is_cnn = 'simplenn';    
        net_simplenn = net;
        clear net;
        
   elseif isequal(features{k}.getFeature, @get_dagnn_layers) 
        if ~isfield(features{k}.fparams, 'input_size_mode')
            features{k}.fparams.input_size_mode = 'adaptive';
        end
        if ~isfield(features{k}.fparams, 'input_size_scale')
            features{k}.fparams.input_size_scale = 1;
        end
        if ~isfield(features{k}.fparams, 'downsample_factor')
            features{k}.fparams.downsample_factor = ones(1, length(features{k}.fparams.output_var));
        end
        
        net = dagnn.DagNN.loadobj(features{k}.fparams.nn_name);
        net.mode = 'test';
        features{k}.fparams.output_var = net.getVarIndex(features{k}.fparams.output_var);
        layer_names = {};
        total_layer_num = numel(net.layers);
        for ii = 1 : total_layer_num - features{k}.fparams.output_var(end)+1
            layer_names{ii} = net.layers(ii + features{k}.fparams.output_var(end)-1).name;
        end
        net.removeLayer(layer_names);
        dim_layer = net.getVarSizes({'data',img_sample_sz(2,:)});
        features{k}.fparams.nDim = zeros(numel(features{k}.fparams.output_var),1);
        for i = 1:numel(features{k}.fparams.output_var)
            features{k}.fparams.nDim(i) = dim_layer{features{k}.fparams.output_var(i)}(3);
        end
        net_RF = net.getVarReceptiveFields(1);
        features{k}.fparams.cell_size = zeros(numel(features{k}.fparams.output_var),1);
        for i = 1:numel(features{k}.fparams.output_var)
             features{k}.fparams.cell_size(i) = net_RF(features{k}.fparams.output_var(i)).stride(1)...
                 * features{k}.fparams.downsample_factor(i);
        end
        features{k}.is_cell = true;
        features{k}.is_cnn = 'dagnn';
        net_dagnn = net;
        clear net;
    else
        error('Unknown feature type');
    end
    
    % Set default cell size
    if ~isfield(features{k}.fparams, 'cell_size')
        features{k}.fparams.cell_size = 1;
    end  
    % Find the minimum cell size of each layer
    feature_info.min_cell_size(k) = min(features{k}.fparams.cell_size);
end

% Order the features in increasing minimal cell size
[~, feat_ind] = sort(feature_info.min_cell_size);
features = features(feat_ind);
feature_info.min_cell_size = feature_info.min_cell_size(feat_ind);

% Set feature info
feature_info.dim_block = cell(num_features,1);
feature_info.img_sample_sz = cell(num_features,1);
feature_info.img_support_sz = cell(num_features,1);

for k = 1:length(features)
    feature_info.dim_block{k} = features{k}.fparams.nDim;
end
% Feature info for each cell block
feature_info.dim = cell2mat(feature_info.dim_block);

% Find if there is any simplenn or dagnn feature
simplenn_feature_ind = -1;
dagnn_feature_ind = -1;
feature_deep_num = 0;
feature_hc_num = 0;
for k = 1:length(features)
    if strcmp(features{k}.is_cnn,'simplenn')
        simplenn_feature_ind = k;
        feature_deep_num = feature_deep_num + 1;
    elseif strcmp(features{k}.is_cnn,'dagnn')
        dagnn_feature_ind = k;
        feature_deep_num = feature_deep_num + 1;
    else
        feature_hc_num = feature_hc_num + 1;
    end
end
feature_info.feature_hc_num = feature_hc_num;
feature_info.feature_deep_num = feature_deep_num;

% if gparams.use_gpu
%     if isempty(gparams.gpu_id)
%         gpuDevice();
%     elseif gparams.gpu_id > 0
%         gpuDevice(gparams.gpu_id);
%     end
% end

if simplenn_feature_ind > 0
    scale = features{simplenn_feature_ind}.fparams.input_size_scale;
    new_img_sample_sz = img_sample_sz(2,:);
    net_info = net_simplenn.info;
    
    if ~strcmpi(size_mode, 'same') && strcmpi(features{simplenn_feature_ind}.fparams.input_size_mode, 'adaptive')
        orig_sz = net_simplenn.info.dataSize(1:2,end)' / features{simplenn_feature_ind}.fparams.downsample_factor(end);
        
        if strcmpi(size_mode, 'exact')
            desired_sz = orig_sz + 1;
        elseif strcmpi(size_mode, 'odd_cells')
            desired_sz = orig_sz + 1 + mod(orig_sz,2);
        end
        
        while desired_sz(1) > net_info.dataSize(1,end)
            new_img_sample_sz = new_img_sample_sz + [1, 0];
            net_info = vl_simplenn_display(net_simplenn, 'inputSize', [round(scale * new_img_sample_sz), 3 1]);
        end
        while desired_sz(2) > net_info.dataSize(2,end)
            new_img_sample_sz = new_img_sample_sz + [0, 1];
            net_info = vl_simplenn_display(net_simplenn, 'inputSize', [round(scale * new_img_sample_sz), 3 1]);
        end
    end
    
    feature_info.img_sample_sz{simplenn_feature_ind} = round(new_img_sample_sz);
    if strcmpi(features{simplenn_feature_ind}.fparams.input_size_mode, 'adaptive')
        features{simplenn_feature_ind}.img_input_sz = feature_info.img_sample_sz{simplenn_feature_ind};
    else
        features{simplenn_feature_ind}.img_input_sz = net.meta.normalization.imageSize(1:2);
    end
    
    % Sample size to be input to the net
    scaled_sample_sz = round(scale * features{simplenn_feature_ind}.img_input_sz);
    
    if isfield(net_info, 'receptiveFieldStride')
        net_info_stride = cat(2, [1; 1], net_info.receptiveFieldStride);
    else
        net_info_stride = [1; 1];
    end
    
    net_stride = net_info_stride(:, features{simplenn_feature_ind}.fparams.output_layer+1)';
    total_feat_sz = net_info.dataSize(1:2, features{simplenn_feature_ind}.fparams.output_layer+1)';
    
    shrink_number = max(2 * ceil((net_stride(end,:) .* total_feat_sz(end,:) - scaled_sample_sz) ./ (2 * net_stride(end,:))), 0);
    
    deepest_layer_sz = total_feat_sz(end,:) - shrink_number;
    scaled_support_sz = net_stride(end,:) .* deepest_layer_sz;
    
    % Calculate output size for each layer
    simple_output_sz = round(bsxfun(@rdivide, scaled_support_sz, net_stride));
    features{simplenn_feature_ind}.fparams.start_ind = floor((total_feat_sz - simple_output_sz)/2) + 1;
    features{simplenn_feature_ind}.fparams.end_ind = features{simplenn_feature_ind}.fparams.start_ind + simple_output_sz - 1;
    
    feature_info.img_support_sz{simplenn_feature_ind} = round(scaled_support_sz .* feature_info.img_sample_sz{simplenn_feature_ind} ./ scaled_sample_sz);
    
    % Set the input size
    net_simplenn.meta.normalization.imageSize(1:2) = round(feature_info.img_sample_sz{simplenn_feature_ind}(1:2));
    net_simplenn.meta.normalization.averageImage = imresize(single(net_simplenn.meta.normalization.averageImageOrig), net_simplenn.meta.normalization.imageSize(1:2));
    net_simplenn.info = vl_simplenn_display(net_simplenn);
    features{simplenn_feature_ind}.fparams.net = net_simplenn;
    
    if gparams.use_gpu
        features{simplenn_feature_ind}.fparams.net = vl_simplenn_move(features{simplenn_feature_ind}.fparams.net, 'gpu');
    end
end

if dagnn_feature_ind > 0
    scale = features{dagnn_feature_ind}.fparams.input_size_scale;
    
    new_img_sample_sz = img_sample_sz(2,:);
    
    if ~strcmpi(size_mode, 'same') && strcmpi(features{dagnn_feature_ind}.fparams.input_size_mode, 'adaptive')
        orig_sz = dim_layer{end}(1:2) / features{dagnn_feature_ind}.fparams.downsample_factor(end);
        
        if strcmpi(size_mode, 'exact')
            desired_sz = orig_sz + 1;
        elseif strcmpi(size_mode, 'odd_cells')
            desired_sz = orig_sz + 1 + mod(orig_sz,2);
        end
        
        while desired_sz(1) > dim_layer{end}(1)
            new_img_sample_sz = new_img_sample_sz + [1, 0];
            dim_layer = net_dagnn.getVarSizes({'data',[floor(scale * new_img_sample_sz), 3 1]});
        end
        while desired_sz(2) > dim_layer{end}(2)
            new_img_sample_sz = new_img_sample_sz + [0, 1];
            dim_layer = net_dagnn.getVarSizes({'data',[floor(scale * new_img_sample_sz), 3 1]});
        end
    end
    
    feature_info.img_sample_sz{dagnn_feature_ind} = round(new_img_sample_sz);
    
    if strcmpi(features{dagnn_feature_ind}.fparams.input_size_mode, 'adaptive')
        features{dagnn_feature_ind}.img_input_sz = feature_info.img_sample_sz{dagnn_feature_ind};
    else
        features{dagnn_feature_ind}.img_input_sz = net_dagnn.meta.normalization.imageSize(1:2);
    end
    
    % Sample size to be input to the net
    scaled_sample_sz = round(scale * features{dagnn_feature_ind}.img_input_sz);
    
    net_stride = net_RF(features{dagnn_feature_ind}.fparams.output_var(end)).stride;
    total_feat_sz = dim_layer{end}(1:2);
    
    shrink_number = max(2 * ceil((net_stride(end,:) .* total_feat_sz(end,:) - scaled_sample_sz) ./ (2 * net_stride(end,:))), 0);
    
    deepest_layer_sz = total_feat_sz(end,:) - shrink_number;
    scaled_support_sz = net_stride(end,:) .* deepest_layer_sz;
    
    % Calculate output size for each layer
    dag_output_sz = round(bsxfun(@rdivide, scaled_support_sz, net_stride));
    features{dagnn_feature_ind}.fparams.start_ind = floor((total_feat_sz - dag_output_sz)/2) + 1;
    features{dagnn_feature_ind}.fparams.end_ind = features{dagnn_feature_ind}.fparams.start_ind + dag_output_sz - 1;
    
    feature_info.img_support_sz{dagnn_feature_ind} = round(scaled_support_sz .* feature_info.img_sample_sz{dagnn_feature_ind} ./ scaled_sample_sz);
    
    % Set the input size
    net_dagnn.meta.normalization.imageSize(1:2) = round(feature_info.img_sample_sz{dagnn_feature_ind}(1:2));
    net_dagnn.meta.normalization.averageImage = imresize(single(net_dagnn.meta.normalization.averageImage), net_dagnn.meta.normalization.imageSize(1:2));
    features{dagnn_feature_ind}.fparams.net = net_dagnn;
    if gparams.use_gpu
        features{dagnn_feature_ind}.fparams.net.move('gpu');
    end
end

for k = 1:length(features)
    if k ~= dagnn_feature_ind && k ~= simplenn_feature_ind
        orig_sz = round(img_sample_sz(1,:)/feature_info.min_cell_size(k));
        if strcmpi(size_mode, 'odd_cells')
            desired_sz = orig_sz + mod(orig_sz,2) +1;
            temp_size = round(img_sample_sz(1,:));
        while desired_sz(1) > temp_size(1)/feature_info.min_cell_size(k)
            temp_size = temp_size + [1, 0];
        end
        while desired_sz(2) > temp_size(2)/feature_info.min_cell_size(k)
            temp_size = temp_size + [0, 1];
        end
        else
            error('Unknown size_mode');
        end
        feature_info.img_support_sz{k} = temp_size;
        feature_info.img_sample_sz{k} = temp_size;  

    end
end
% Set the sample size and data size for each feature
feature_info.data_sz_block = cell(num_features,1);
feature_info.learning_rate_block = cell(num_features,1);
feature_info.channel_selection_rate_block = cell(num_features,1);
feature_info.spatial_selection_rate_block = cell(num_features,1);
feature_info.feature_is_deep_block = cell(num_features,1);
for k = 1:length(features)
    if strcmp(features{k}.is_cnn,'simplenn') 
        % CNN features have a different sample size, since the receptive
        % field is often larger than the support size
        features{k}.img_sample_sz = feature_info.img_sample_sz{k};
        
        % Set the data size based on the computed output size
        feature_info.data_sz_block{k} = floor(bsxfun(@rdivide, simple_output_sz, features{k}.fparams.downsample_factor'));
        feature_info.learning_rate_block{k} = repmat(params.learning_rate(features{k}.fparams.feature_is_deep+1),numel(features{k}.fparams.output_layer),1);
        feature_info.channel_selection_rate_block{k} = repmat(params.channel_selection_rate(features{k}.fparams.feature_is_deep+1),numel(features{k}.fparams.output_layer),1);
        feature_info.spatial_selection_rate_block{k} = repmat(params.spatial_selection_rate(features{k}.fparams.feature_is_deep+1),numel(features{k}.fparams.output_layer),1);
        feature_info.feature_is_deep_block{k} = repmat(features{k}.fparams.feature_is_deep,numel(features{k}.fparams.output_layer),1);
    elseif strcmp(features{k}.is_cnn,'dagnn')
        features{k}.img_sample_sz = feature_info.img_sample_sz{k};
        
        % Set the data size based on the computed output size
        feature_info.data_sz_block{k} = floor(bsxfun(@rdivide, dag_output_sz, features{k}.fparams.downsample_factor'));
        feature_info.learning_rate_block{k} = repmat(params.learning_rate(features{k}.fparams.feature_is_deep+1),numel(features{k}.fparams.output_var),1);
        feature_info.channel_selection_rate_block{k} = repmat(params.channel_selection_rate(features{k}.fparams.feature_is_deep+1),numel(features{k}.fparams.output_var),1);
        feature_info.spatial_selection_rate_block{k} = repmat(params.spatial_selection_rate(features{k}.fparams.feature_is_deep+1),numel(features{k}.fparams.output_var),1);
        feature_info.feature_is_deep_block{k} = repmat(features{k}.fparams.feature_is_deep,numel(features{k}.fparams.output_var),1);
    else
        % implemented classic features always have the same sample and
        % support size
        features{k}.img_sample_sz = feature_info.img_support_sz{k};
        features{k}.img_input_sz = features{k}.img_sample_sz;
        
        % Set data size based on cell size
        feature_info.data_sz_block{k} = floor(bsxfun(@rdivide, features{k}.img_sample_sz, features{k}.fparams.cell_size));
        feature_info.learning_rate_block{k} = params.learning_rate(features{k}.fparams.feature_is_deep+1);
        feature_info.channel_selection_rate_block{k} = params.channel_selection_rate(features{k}.fparams.feature_is_deep+1);
        feature_info.spatial_selection_rate_block{k} = params.spatial_selection_rate(features{k}.fparams.feature_is_deep+1);
        feature_info.feature_is_deep_block{k} = features{k}.fparams.feature_is_deep;
    end
end

feature_info.data_sz = cell2mat(feature_info.data_sz_block);
feature_info.learning_rate = cell2mat(feature_info.learning_rate_block);
feature_info.channel_selection_rate = cell2mat(feature_info.channel_selection_rate_block);
feature_info.spatial_selection_rate = cell2mat(feature_info.spatial_selection_rate_block);
feature_info.feature_is_deep = cell2mat(feature_info.feature_is_deep_block);