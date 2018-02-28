# Adaptation Regularization based Transfer Learning
<hr>


**Method Description**

Another method is called Adaptation Regularization based Transfer Learning ARTL it falls within the family of unsupervised domain adaptation techniques. It combines different strategies for transfer learning within a single framework: it simultaneously optimizes the structural risk functional over the source domain, the joint distribution matching of both marginal and conditional distributions, and the consistency of the geometric manifold corresponding to the marginal distribution.

**How To Use**

```Matlab
ARTL(SrcX, SrcY, TgtX, TgtY, OptionsARTL)

% SrcX: Source dataset predictors.
% SrcY: Source dataset responses.
% TgtX: Target dataset predisctors.
% TgtY: Target dataset responses.
$ OptionsARTL: Structure containing p, 
% 			   sigma, lambda, gamma, 
%              kernel_type, and T values. 
%
% Refer to the reference for more information on these values.
```

**Reference**

[Adaptation Regularization]()