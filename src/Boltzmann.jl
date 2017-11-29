
module Boltzmann

export AbstractRBM,
       RBM,
       BernoulliRBM,
       GRBM,
       ConditionalRBM,
       IsingRBM,
       DBN,
       DAE,
       Bernoulli,
       Gaussian,
       Degenerate,
       fit,
       transform,
       generate,
       predict,
       coef,
       weights,
       hbias,
       vbias,
       unroll,
       save_params,
       load_params,
       persistent_meanfield,
       sampler_UpdMeanfield,
       iter_mag

include("core.jl")

end
