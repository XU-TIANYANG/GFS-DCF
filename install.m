% Compile libraries, download network modles and demo sequences for GFS-DCF
[path_root, name, ext] = fileparts(mfilename('fullpath'));

% mtimesx
if exist('tracker_exter/mtimesx', 'dir') == 7
    cd tracker_exter/mtimesx
    mtimesx_build;
    cd(path_root)
end

% PDollar toolbox
if exist('tracker_exter/pdollar_toolbox/external', 'dir') == 7
    cd tracker_exter/pdollar_toolbox/external
    toolboxCompile;
    cd(path_root)
end

% matconvnet
if exist('tracker_exter/matconvnet/matlab', 'dir') == 7
    cd tracker_exter/matconvnet/matlab
    vl_compilenn; % enable/disable GPU based on your hardware
    cd(path_root)
    
    % donwload network
    cd tracker_featu
    mkdir offline_models
    cd offline_models
    if ~(exist('imagenet-resnet-50-dag.mat', 'file') == 2)
        disp('Downloading the network "imagenet-resnet-50-dag.mat" from "http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat"...')
        urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat', 'imagenet-resnet-50-dag.mat');
        disp('Done!')
    end
    cd(path_root)
else
    error('GFS-DCF : Matconvnet not found.')
end

% download demo sequences
if exist('tracker_seque', 'dir') == 0
    mkdir tracker_seque
    cd tracker_seque
else 
    cd tracker_seque
end
if exist('Biker', 'dir') == 0
    disp('Downloading the demo sequence (1/2) "Biker" from "http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/Biker.zip"...')
    urlwrite('http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/Biker.zip', 'Biker.zip');
    unzip('Biker.zip');
    rmdir('__MACOSX','s');
    delete('Biker.zip');
    disp('Done!')
end
if exist('bag', 'dir') == 0   
    mkdir bag
    cd bag
    mkdir color
    cd color
    disp('Downloading the demo sequence (2/2) "Bag" from "http://data.votchallenge.net/sequences/28b56d282ad4abeaaca820b1bebcab2f2aeb7a9a8b0da71f1103f7853a0add7e80ea7d6030892616d0ca0a639366418d3587c5e55bf759be91c4cfb514d53751.zip"...')
    urlwrite('http://data.votchallenge.net/sequences/28b56d282ad4abeaaca820b1bebcab2f2aeb7a9a8b0da71f1103f7853a0add7e80ea7d6030892616d0ca0a639366418d3587c5e55bf759be91c4cfb514d53751.zip', 'bag.zip');
    unzip('bag.zip');
    delete('bag.zip'); 
    cd ..
    urlwrite('http://data.votchallenge.net/vot2018/main/bag.zip', 'anno.zip');
    unzip('anno.zip');
    delete('anno.zip'); 
    cd ..
    disp('Done!')
end
cd(path_root)
    