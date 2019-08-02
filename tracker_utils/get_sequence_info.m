function [seq, init_image] = get_sequence_info(seq)

if ~isfield(seq, 'format') || isempty(seq.format)
    if isempty(seq)
        seq.format = 'vot';
    else
        seq.format = 'otb';
    end
else
    if strcmpi(seq.format, 'rect_4')
        seq.format = 'otb';
    elseif strcmpi(seq.format, 'rect_8')
        seq.format = 'otb_8';
    else
    error('Uknown sequence format');
    end
end

seq.frame = 0;

if strcmpi(seq.format, 'otb')||strcmpi(seq.format, 'otb_8')
    if numel(seq.init_rect) > 4
        seq.rect8 = round(seq.init_rect(:));
        rect8 = seq.rect8;
        x1 = round(min(rect8(1:2:end)));
        x2 = round(max(rect8(1:2:end)));
        y1 = round(min(rect8(2:2:end)));
        y2 = round(max(rect8(2:2:end)));
        seq.init_rect = round([x1, y1, x2 - x1, y2 - y1]);
        seq.target_mask = single(poly2mask(rect8(1:2:end)-seq.init_rect(1), ...
        rect8(2:2:end)-seq.init_rect(2), seq.init_rect(4), seq.init_rect(3)));
        seq.t_b_ratio = sum(seq.target_mask(:))/prod(seq.init_rect([4,3]));
    else
        r = seq.init_rect(:);
        seq.rect8 = [r(1),r(2),r(1)+r(3),r(2),r(1)+r(3),r(2)+r(4),r(1),r(2)+r(4)];
        seq.target_mask = single(ones(seq.init_rect([4,3])));
        seq.t_b_ratio = 1;
    end
    seq.image_files = seq.s_frames;
    seq = rmfield(seq, 's_frames');
    seq.init_sz = [seq.init_rect(1,4), seq.init_rect(1,3)];
    seq.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + (seq.init_sz - 1)/2;
    seq.num_frames = numel(seq.image_files);
    seq.rect_position = zeros(seq.num_frames, 4);
    init_image = imread(seq.image_files{1});
elseif strcmpi(seq.format, 'vot')
    [seq.handle, init_image_file, init_region] = vot('polygon');
    
    if isempty(init_image_file)
        init_image = [];
        return;
    end
    
    init_image = imread(init_image_file);
    
    bb_scale = 1;
    
    if numel(init_region) > 4
        cx = mean(init_region(1:2:end));
        cy = mean(init_region(2:2:end));
        x1 = min(init_region(1:2:end));
        x2 = max(init_region(1:2:end));
        y1 = min(init_region(2:2:end));
        y2 = max(init_region(2:2:end));
        A1 = norm(init_region(1:2) - init_region(3:4)) * norm(init_region(3:4) - init_region(5:6));
        A2 = (x2 - x1) * (y2 - y1);
        s = sqrt(A1/A2);
        w = s * (x2 - x1) + 1;
        h = s * (y2 - y1) + 1;
    else
        cx = init_region(1) + (init_region(3) - 1)/2;
        cy = init_region(2) + (init_region(4) - 1)/2;
        w = init_region(3);
        h = init_region(4);
    end
    
    init_c = [cy cx];
    init_sz = bb_scale * [h w];
    
    im_size = size(init_image);
    
    seq.init_pos = init_c;
    seq.init_sz = min(max(round(init_sz), [1 1]), im_size(1:2));
    seq.num_frames = Inf;
    seq.region = init_region;
end