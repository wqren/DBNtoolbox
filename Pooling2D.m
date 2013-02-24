classdef Pooling2D < handle & NetworkLayer
	properties
		ws;       %window size
        stride;   
                
        type;        
        
        max_idx; %for record the index of max-pooling
	end
	
	methods								
        function self = Pooling2D(ws, stride, type)            
            self.ws = ws;
            self.stride = stride;
            self.type = type;
        end
        
        function [] = setPar(self,in_size)					
            self.in_size = in_size;            
            new_size = [(self.in_size(1)-self.ws)/self.stride+1,(self.in_size(2)-self.ws)/self.stride+1];
            if nnz(mod(new_size,1))> 0
               error('pooling dimension mismatch');
            end
            self.out_size = [new_size, in_size(3)];            
        end
        
        function object = gradCheckObject(self)                        
            ws = 3;
            stride = 2;
            object = Pooling2D(ws, stride,self.type);
        end  
        		
		function [] = fprop(self)            
			self.OUT = zeros([(self.in_size(1)-self.ws)/self.stride+1, (self.in_size(2)-self.ws)/self.stride+1, self.in_size(3),self.numdata]);            
            if strcmp(self.type,'max')
               self.OUT = -Inf*(self.OUT+1);           
               self.max_idx = zeros(size(self.OUT));
            end
            
            
            for i = 1 : self.ws %it's usually smaller
                for j = 1 : self.ws
                    tmp = self.IN(i:self.stride:end-self.ws+i, j:self.stride:end-self.ws+j,:,:);
                    switch self.type
                        case 'max'
                            mask = tmp > self.OUT;
                            self.max_idx(mask) = (i-1)*self.ws+j;
                            self.OUT(mask) = tmp(mask);
                        case 'avg'
                            self.OUT = self.OUT + tmp;
                        otherwise
                            error('non-implemented type');
                    end                    
                end
            end
            if strcmp(self.type,'avg')
                self.OUT = self.OUT / (self.ws^2);
            end                        
		end
        
		function [OUT] = fprop_rev(self, IN)
		%for computing the gradient in ConvAutoencoder
			OUT = zeros([(self.in_size(1)-self.ws)/self.stride+1, (self.in_size(2)-self.ws)/self.stride+1, self.in_size(3),self.numdata]);
            if nnz(mod(OUT,1))>0
               error('pooling dimension mismatch');
            end
            
            for i = 1 : self.ws %it's usually smaller
                for j = 1 : self.ws
                    tmp = IN(i:self.stride:end-self.ws+i, j:self.stride:end-self.ws+j,:,:);
                    switch self.type
                        case 'max'                            
                            idx = self.max_idx == (i-1)*self.ws+j;
                            OUT(idx) = tmp(idx);
                        case 'avg'
                            OUT = OUT + tmp;
                        otherwise
                            error('non-implemented type');
                    end                    
                end
            end
            if strcmp(self.type,'avg')
                OUT = OUT / (self.ws^2);
            end     
		end
		
		function [f derivative] = bprop(self,f,derivative)            
            if ~self.skip_passdown                
                dX = zeros([self.in_size self.numdata]);                
                
                for i = 1 : self.ws
                    for j = 1 : self.ws
                        switch self.type
                            case 'max'
                                dX(i:self.stride:end-self.ws+i, j:self.stride:end-self.ws+j,:,:) = ...
                                    dX(i:self.stride:end-self.ws+i, j:self.stride:end-self.ws+j,:,:) + derivative .* (self.max_idx == (i-1)*self.ws+j);
                            case 'avg'
                                dX(i:self.stride:end-self.ws+i, j:self.stride:end-self.ws+j,:,:) = ...
                                    dX(i:self.stride:end-self.ws+i, j:self.stride:end-self.ws+j,:,:)+derivative;
                        end
                    end
                end
                
                if strcmp(self.type,'avg')
                    dX = dX / (self.ws^2);
                end
                derivative = dX;
            end
		end
	end
end