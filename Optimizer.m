classdef Optimizer < handle
%for optimization, default is minFunc
    properties
        par;
    end
    
    methods 
        function [x] = run(self, funObj, x0)
            addpath(genpath(fullfile(Config.lib_dir_path,'minFunc')));
            x = minFunc(funObj,x0,self.par);            
        end        
    end

    methods(Static)
    end
end