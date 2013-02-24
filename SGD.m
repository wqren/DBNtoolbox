classdef SGD < handle & Optimizer
    properties
        init_eps = 1e-5; %initial learning rate
        eps_par; %parameters for updating learning rate
        eps_update; %function handle for update learning rate
        
        init_momentum = 0.5; %initial momentum
        final_momentum = 0.99; %final momentum 
        momentum_par; 
        momentum_update;
        
        max_iter = 1;                

        %temporary variable
        xinc;
        curr_t = 1;
    end
    
    methods
        function [self] = SGD()
            self.eps_update = @(t)self.expDecay(t);            
%             self.momentum_update = @(t)self.linearWeight(t);        
%             self.eps_update = @(t)self.constantEps(t);
            self.momentum_update = @(t)self.constantMomentum(t);
        end
        
        function [x] = run(self, funObj, x0) %update with all samples            
            if isempty(self.xinc)
                self.xinc = Utils.zeros(size(x0));
            end
            
            for t = self.curr_t:self.curr_t+self.max_iter-1
                eps = self.eps_update(t);
                m = self.momentum_update(t);                 
                [f g] = funObj(x0);
                self.xinc = m*self.xinc - eps*g;
                x = x0 + self.xinc;
                fprintf('iter: %d, fobj: %f, |x|: %f\n',t, f, norm(x));
            end
            self.curr_t = self.curr_t+self.max_iter;
        end
        
        %function [x] = runMiniBatch(self, funObj, x0, X, Y, par)
%            x = x0;
%            xinc = x;
%            numsamples = size(X,2);            
%            randidx = randperm(numsamples);
%            
%            for t = 1 : self.max_iter
%                eps = self.eps_update(t);
%                m = self.momentum_update(t); 
%                for b = 1 : floor(numsamples/self.batch_size)
%                    batchidx = randidx((b-1)*self.batch_size+1:b*self.batch_size);
%                    
%                    [f g] = funObj(x, X(:,batchidx), Y(batchidx));
%                    x = m*xinc - (1-m)*eps*g;
%                end
%            end
%        end
        %------------eps update---------
        function [eps] = expDecay(self,t)
            if isempty(self.eps_par)
                f = 0.998;
            else
                f = self.eps_par(1);
            end 
            
            eps = self.init_eps*(f^t);
        end
        
        function [eps] = constantEps(self,t)
            eps = self.init_eps;
        end
        
        %-----momentum update--------
        function [momentum] = constantMomentum(self,t)
            momentum = self.init_momentum;
        end
        
        function [momentum] = twoPhase(self,t)
            if t > self.momentum_par(1)                
                momentum = self.init_momentum;
            else
                momentum = self.final_momentum;
            end
        end
                
        function [momentum] = linearWeight(self,t)
            if isempty(self.momentum_par)
                stop_epoch = 500;
            else
                stop_epoch = self.momentum_par(1);
            end
            
            if t > stop_epoch
                momentum = self.final_momentum;
            else
                momentum = (t/stop_epoch) * self.final_momentum + (1-t/stop_epoch) * self.init_momentum; 
            end
        end
    end
    
    methods(Static)
        
    end
end