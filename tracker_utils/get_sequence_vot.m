function seq = get_sequence_vot(seq)
if strcmpi(seq.format, 'vot')
    if numel(seq.region) > 4
        seq.rect8 = round(seq.region(:));
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
        r = seq.region(:);
        seq.rect8 = [r(1),r(2),r(1)+r(3),r(2),r(1)+r(3),r(2)+r(4),r(1),r(2)+r(4)];
        seq.target_mask = single(ones(seq.region([4,3])));
        seq.t_b_ratio = 1;
    end
end
end

