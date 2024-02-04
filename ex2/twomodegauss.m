function p=twomodegauss(m1, sig1, m2, sig2, A1, A2, k)
% Generating binomial distribution. A good starting point could be 
% distribution with the following arguments:
% twomodegauss(0.15, 0.05, 0.75, 0.05, 1, 0.07, 0.002)
c1 = A1 * (1 / ((2 * pi) ^ 0.5) * sig1);
k1 = 2 * (sig1 ^ 2);
c2 = A2 * (1 / ((2 * pi) ^ 0.5) * sig2);
k2 = 2 * (sig2 ^ 2);
z = linspace(0, 1, 256);
p = k + c1 * exp(-((z - m1) .^ 2) ./ k1) + ...
    c2 * exp(-((z - m2) .^ 2) ./ k2);
p = p ./ sum(p(:));