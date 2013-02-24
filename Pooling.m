classdef Pooling < handle
%Not implemented yet
%Pooling layer for general (feadim*numdata) data, for two dimensional data like image, see Pooling2D    
    properties
        type
        
        %as component in DBN        
        OUT;        
    end
    
    methods
        function self = Pooling(type)
            self.type = type;
        end
        
        function [Xout] = fprop(self,X) %X = feadim * numchannels * numsamples
            switch self.type
                case 'max'
                    Xout = squeeze(max(X,[],2));
                case 'avg'
                    Xout = squeeze(mean(X,2));                
                otherwise 
                    error('type for pooling not supported');
            end
            self.OUT = Xout;
        end
    end

end