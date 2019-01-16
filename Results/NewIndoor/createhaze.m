clc;
clear;
fileFolder=fullfile('C:\Users\liuz156\Desktop\indoor\depth');
dirOutput=dir(fullfile(fileFolder,'*.png'));
fileNames={dirOutput.name};
len = length(fileNames);
for i = 1 : len    
    depth_name = strcat('C:\Users\liuz156\Desktop\indoor\depth\', fileNames{i}) ;
    tokens = strsplit(fileNames{i}, '.');
    rbg_name = strcat('C:\Users\liuz156\Desktop\indoor\rgb\', tokens{1}, '.jpg');
    % Beta is in the range of 0.2 to 0.4
    % Alpha is in the range of 0.8 to 1   
    beta = 0.2 + randsample(10, 1) * 0.02;
    alpha = 0.8 + randsample(10, 1) * 0.02;
    depthImg = double(imread(depth_name));
    depth = depthImg/ 10000;
    trans = exp(-1 * beta * depth);
    rgbImg = imread(rbg_name);
    rgbImg = im2double(rgbImg);
    [h, w, c] = size(rgbImg);
    haze = zeros(h, w, 3);
    A = ones(h, w) * alpha;
    for j = 1:3
       haze(:, :, j)  = rgbImg(:,:,j) .* trans + A .* (ones(h, w) - trans);
    end
    imwrite(haze, strcat('.\result\', tokens{1}, '_', num2str(beta), '_', num2str(alpha), '.png'));
end
