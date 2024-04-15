function [ratio,RMSE,Fidelity] = Fx_evaluation(field, target, source)
shielder = zeros(size(target));
shielder(target > 0.0) = 1;


ener0 = sum(abs(source).^2,'all');
ener1 = sum(abs(field).^2 .* shielder,'all');
ratio = ener1 / ener0

enert = sum(abs(target).^2 .* shielder,'all');
enert = sqrt(enert ./ sum(shielder,'all'));
ener1 = sqrt(ener1 ./ sum(shielder,'all'));


RMSE = sqrt(sum((abs(field ./ ener1).^2 - abs(target ./ enert).^2).^2 .* shielder,'all') / sum(shielder,'all'))


Fidelity = sum(abs(field ./ ener1) .* abs(target ./ enert), "all") ./ (sum(abs(field ./ ener1).^2,"all") * sum(abs(target ./ enert).^2,"all")).^0.5;
Fidelity = Fidelity.^2
end