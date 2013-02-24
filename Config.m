classdef Config < handle
    properties (Constant = true)
        lib_dir_path = 'C:\Users\mysu\code\lib';
        basis_dir_path = 'C:\Users\mysu\code\basis';
        fea_dir_path = '';
        data_dir_path = '\\david\DOAdvanced\Qumulus_workspaces\mysu';
        tmp_dir_path = '';
        result_dir_path = '';
        
        gpu = false;       
    end
end