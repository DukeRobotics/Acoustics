
function Bearing = GiveBearing(phase0, phase1, phase2)
    %% Matlab processing

    c=1550; %meters/second
    x=0.028575; %meters, 9/8" 

    diff10 = phase1 - phase0;
    diff20= phase2-phase0;
    if diff10> pi
        diff10=diff10-2*pi
    elseif diff10<-pi
        diff10=diff10+2*pi
    end
    if diff20> pi
        diff20=diff20-2*pi
    elseif diff20<-pi
        diff20=diff20+2*pi
    end

   Bearing=atan((phase1-phase0)./(phase2-phase0));
   %Bearing=atan2(diff31,diff21);


end
