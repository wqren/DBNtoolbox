classdef GroupSparsity < handle & NetworkLayer
    %adding group sparsity (only implement 1D topography)
    %try to complement all data format at the same time, do sparsity on dimension right before "numdata"
    %configurable for passing result up or not
	properties
		kernel; %gaussian kernel
        stride; %stride when applying kernel
        ws;     %width of the kernel
        lambda = 0.1; %sparsity weight
        ws_rad = 4; %std of the kernel        
        pass_up = false; %pass the result to upper layer
        numunits; %number of pooling units
        target_sparsity = 0; %when larger than 0, use KL sparsity penalty
        
        link_map; %temporary variable for saving linkage map, it's a speed/memory tradeoff
        pooled; %temporary variable for saving pooled output
        epsilon = 1e-5; %small number to prevent numerical error
	end
	
	methods
        function self = GroupSparsity(ws, stride)
            self.ws = ws;
            self.stride = stride;
            self.kernel = fspecial('gaussian',self.ws,self.ws/self.ws_rad);
            self.kernel = self.kernel(:,round(self.ws/2));
            self.kernel = self.kernel/sum(self.kernel);
        end
        		
		function [] = setPar(self,in_size)		
            self.in_size = in_size;
            self.out_size = in_size;
            self.numunits = (in_size(end)-self.ws)/self.stride + 1;
            if mod(self.numunits,1) ~= 0
                error('pooling unit mismatch in GroupSparsity layer');
            end
            if self.pass_up
                self.out_size(end) = self.numunits;%after pooling
            end
            
            self.link_map = Utils.zeros([in_size(end),self.numunits]); %again this is memory intensive
            span = 1:self.ws;
            for i = 1 : self.numunits;
                self.link_map(span,i) = self.kernel;
                span = span + self.stride;
            end            
		end
		        		        				                		
        function object = gradCheckObject(self)                        
            ws = 2;
            stride = 1;
            object = GroupSparsity(ws,stride);            
            object.lambda = 1000;
            object.target_sparsity = self.target_sparsity;
        end  
        
        function clearTempData(self)           
            self.clearTempData@NetworkLayer();
            self.pooled = [];
        end   
        
		%all layers must implement these
		function [] = fprop(self)            
%             if self.ws == 1 %simple sparsity
%                 self.OUT = self.IN;
%                 self.pooled = self.IN;
%                 return;
%             end
            
            tmp = self.IN;
            if length(self.in_size)>1
                tmp = permute(tmp,[length(self.in_size), 1:(length(self.in_size)-1), length(self.in_size)+1]);
                tmp = reshape(tmp, [self.in_size(end), prod(self.in_size(1:end-1))*self.numdata]);
            end
            tmp = tmp.^2;
            self.pooled = sqrt(self.link_map'*tmp+self.epsilon);
			
            if self.pass_up                                                
                self.OUT = self.pooled;
                if length(self.in_size) > 1
                    self.OUT = reshape(self.OUT, [self.out_size(end), self.in_size(1:end-1), self.numdata]);
                    self.OUT = permute(self.OUT, [2:length(self.in_size), 1, length(self.in_size)+1]);
                end                 
            else
                self.OUT = self.IN;
            end
		end
        
		function [f derivative] = bprop(self,f,derivative)                        
            if self.target_sparsity > 0                
                curr_sparsity = mean(self.pooled,2);
                tmp1 = self.target_sparsity./curr_sparsity;
                tmp2 = (1-self.target_sparsity)./(1-curr_sparsity);
                f = f + self.lambda*size(self.pooled,2)*(sum(self.target_sparsity*log(tmp1) + (1-self.target_sparsity)*log(tmp2)));
                dX = repmat((-tmp1+tmp2),[1 size(self.pooled,2)]);
            else
                f = f + self.lambda*sum(self.pooled(:));
                dX = Utils.ones(size(self.pooled));
            end
                                        
            if self.pass_up
                if length(self.in_size)>1
                    derivative = permute(derivative,[length(self.in_size), 1:(length(self.in_size)-1), length(self.in_size)+1]);
                    derivative = reshape(derivative, [self.out_size(end), prod(self.out_size(1:end-1))*self.numdata]);
                end
                dX = derivative + self.lambda*dX;
            end
            
            dX = self.link_map*(dX./self.pooled);
            
            if length(self.in_size) > 1
                dX = reshape(dX, [self.in_size(end), self.in_size(1:end-1), self.numdata]);
                dX = permute(dX, [2:length(self.in_size), 1, length(self.in_size)+1]);
            end
            dX = dX .* self.IN;

            if ~self.pass_up
                derivative = derivative + self.lambda*dX;
            else
                derivative = dX;
            end
                
            if self.skip_passdown %compute gradient for lower layers
                error('sparsity should always pass down gradient')
            end			
		end
	end
end