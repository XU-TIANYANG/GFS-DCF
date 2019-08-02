function [seq, ground_truth] = load_video(video_path,sequence_name)
% Load the information of 'sequence_name' in 'video_path'.
% Input  : sequence_name[str](sequence name)
%          video_path[str](sequence path)
% Output : seq[structure](sequence information)
%          ground_truth[matrix](ground truth bounding boxes)
% [1] Wu Y, Lim J, Yang MH. Object tracking benchmark. IEEE Transactions on 
% Pattern Analysis and Machine Intelligence. 2015 Sep 1;37(9):1834-48.

% set sequence format (4-parameter or 8-parameter bounding box)
if exist([video_path '/groundtruth_rect.txt'])
    ground_truth = dlmread([video_path '/groundtruth_rect.txt']);
    seq.format = 'rect_4';
elseif exist([video_path '/groundtruth.txt'])
    ground_truth = dlmread([video_path '/groundtruth.txt']);
    seq.format = 'rect_8';
end

seq.len = size(ground_truth, 1);
seq.init_rect = ground_truth(1,:);

% set start and end frame based on [1]
if strcmp(sequence_name, 'David')
    start_frame = 300;end_frame = 770;
elseif strcmp(sequence_name, 'Football1')
    start_frame = 1;end_frame = 74;
elseif strcmp(sequence_name, 'Freeman3')
    start_frame = 1;end_frame = 460;
elseif strcmp(sequence_name, 'Freeman4')
    start_frame = 1;end_frame = 283;
else
    start_frame = 1; end_frame = seq.len;
end

% set frame index 
if strcmp(sequence_name, 'BlurCar1')
    nn = 247;
elseif strcmp(sequence_name, 'BlurCar3')
    nn = 3;
elseif strcmp(sequence_name, 'BlurCar4')
    nn = 18;
else
    nn = 1;
end

% set image path
switch seq.format
    case 'rect_4'
        img_path = [video_path '/img/'];
        if exist([img_path num2str(nn, '%04i.png')], 'file')
            img_files = num2str((start_frame+nn-1:end_frame+nn-1)', [img_path '%04i.png']);
        elseif exist([img_path num2str(nn, '%04i.jpg')], 'file')
            img_files = num2str((start_frame+nn-1:end_frame+nn-1)', [img_path '%04i.jpg']);
        elseif exist([img_path num2str(nn, '%04i.bmp')], 'file')
            img_files = num2str((start_frame+nn-1:end_frame+nn-1)', [img_path '%04i.bmp']);
        elseif exist([img_path num2str(nn, '%05i.jpg')], 'file')
            img_files = num2str((start_frame+nn-1:end_frame+nn-1)', [img_path '%05i.jpg']);
        else
            error('No image files to load.')
        end
    case 'rect_8'
        img_path = [video_path '/color/'];
        img_files = num2str((start_frame:end_frame)', [img_path '%08i.jpg']);
end
seq.s_frames = cellstr(img_files);
seq.start_frame = start_frame;
seq.end_frame = end_frame;
end

