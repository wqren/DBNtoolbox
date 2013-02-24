classdef Learner < handle
    properties
        M; %mean, for normalization
        S; %std, for normalization
        P; %for whitening, to whiten data -> P*X
        
        %training variables
        max_iter = 100;
        save_iter;
        save_dir;
    end
    
    methods
        function self = Learner()
        end
        
        function [] = visualize1D(self, savepath, disp_col)
            if ~exist('savepath','var')
                savepath = './tmp';
            end
            clf;
            if ~exist('disp_col','var')            
                disp_col = round(sqrt(self.numunits));
            end
            disp_row = self.numunits / disp_col;
            for i = 1 : self.numunits
                subplot(disp_col,disp_row,i);
                plot(self.weights(:,i));
            end
            saveas(gcf,[savepath '.png']);
        end
    end
end