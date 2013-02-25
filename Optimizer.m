classdef Optimizer < handle
%for optimization, default is minFunc
    properties
        par;
    
        batch_size = 100;
        max_iter = 100;
    end
    
    methods 
        function [x] = run(self, funObj, x0)
            %handle parameters only
            addpath(genpath(fullfile(Config.lib_dir_path,'minFunc')));
            x = minFunc(funObj,x0,self.par);            
        end        
        
        function [] = batchUpdate(self, obj, X)                            
            numdata = size(X,2); %add more choices later
            numbatch = floor(numdata / self.batch_size);
            
            obj.initialization(X);
            for t = 1 : self.max_iter
                fprintf('iter: %d,',t);
                obj.initIter(t);
                randidx = randperm(numdata);
                for b = 1 : numbatch
                    Xbatch = X(:, randidx( (b-1)*self.batch_size+1:b*self.batch_size ));
                    obj.update(Xbatch);
                end
                %TODO: save if needed here
                if obj.checkStop()
                    break;
                end
            end        
        end 
        
    end

    methods(Static)
           
    end
end