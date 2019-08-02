function video = choose_video(base_path)
% Choose a specific video from 'base_path'.
% Input  : base_path[str](root path of a sequence set)
% Output : video[str](selected folder name)

%process suitable path for system
if ispc(), base_path = strrep(base_path, '\', '/'); end
if base_path(end) ~= '/', base_path(end+1) = '/'; end

%list all sub-folders
contents = dir(base_path);
names = {};
for k = 1:numel(contents)
    name = contents(k).name;
    if isdir([base_path name]) && ~strcmp(name, '.') && ~strcmp(name, '..')
        names{end+1} = name;  %#ok
    end
end

%no sub-folders found
if isempty(names), video = []; return; end

%choice GUI
choice = listdlg('ListString',names, 'Name','Choose video', 'SelectionMode','single');

if isempty(choice) 
    video = [];
else
    video = names{choice};
end

end

