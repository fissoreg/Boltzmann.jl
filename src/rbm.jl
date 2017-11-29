# TODO : correcting the negative term


using Base.LinAlg
using Base.LinAlg.BLAS
using Distributions
import StatsBase.fit
import StatsBase.coef
import StatsBase: sample, sample!

@runonce const Mat{T} = AbstractArray{T, 2}
@runonce const Vec{T} = AbstractArray{T, 1}
@runonce const Gaussian = Distributions.Normal
@runonce const IsingSpin = Distributions.Categorical

"""
Distribution with a single possible value. Used e.g. during sampling
to provide stable result equal to provided means:

sample(Degenerate, means) = means
"""
struct Degenerate <: Distribution{Distributions.Univariate,
                               Distributions.Discrete}
end


## types


@runonce begin

    # TODO: why these dosctrings are not attached to these types?
    """
    Base type for all RBMs. Takes type parameters:

     * T - type of RBM parameters (weights, biases, etc.), input and output.
       By default, Float64 is used
     * V - type of visible units
     * H - type of hidden units
    """
    abstract type AbstractRBM{T,V,H} end

    """
    Restricted Boltzmann Machine, parametrized by element type T, visible
    unit type V and hidden unit type H.
    """
    mutable struct RBM{T,V,H} <: AbstractRBM{T,V,H}
        W::Matrix{T}         # matrix of weights   between vis and hid vars
        W2::Matrix{T}        # matrix of weights^2 between vis and hid vars
        vbias::Vector{T}     # biases for visible variables
        hbias::Vector{T}     # biases for hidden variables
	    activation::Function
        means::Function
        marg::Function
        flip::Function
    end

end

"""
Construct RBM. Parameters:

 * T - type of RBM parameters (e.g. weights and biases; by default, Float64)
 * V - type of visible units
 * H - type of hidden units
 * activation - activation function to use
 * n_vis - number of visible units
 * n_hid - number of hidden units

Optional parameters:

 * sigma - variance to use during parameter initialization

"""
function RBM(T::Type, V::Type, H::Type, activation::Function, means::Function, marg::Function,
             flip::Function, n_vis::Int, n_hid::Int; sigma=0.01, InitVisBias=zeros(n_vis))
    RBM{T,V,H}(map(T, rand(Normal(0, sigma), n_hid, n_vis)), zeros(n_hid, n_vis),
             InitVisBias, zeros(n_hid), activation, means, marg, flip)
end

RBM(V::Type, H::Type, n_vis::Int, n_hid::Int; sigma=0.01, InitVisBias=zeros(n_vis)) =
    RBM(Float64, V, H, logistic, MeansBernoulli, MargHiddenBernoulli, FlipBernoulli, n_vis, n_hid; sigma=sigma, InitVisBias=InitVisBias)


# some well-known RBM kinds

"""Same as RBM{Float64,Degenerate,Bernoulli}"""
function BernoulliRBM(n_vis::Int, n_hid::Int; sigma=0.01, TrainData=[])
    InitialVisBias = zeros(n_vis)    
    if !isempty(TrainData)
        InitialVisBias = getBiasFromSamples(TrainData, 1.0)
    end
    return RBM(Degenerate, Bernoulli, n_vis, n_hid; sigma=sigma,  InitVisBias=InitialVisBias[:,1])
end

"""Same as RBM{Float64,Gaussian,Bernoulli}"""
GRBM(n_vis::Int, n_hid::Int; sigma=0.01) =
    RBM(Normal, Bernoulli, n_vis, n_hid; sigma=sigma)

"""Same as IsingRBM{Float64,Ising,Ising}"""
function IsingRBM(n_vis::Int, n_hid::Int; sigma=0.01, TrainData=[])

    InitialVisBias = zeros(n_vis)
    if !isempty(TrainData)
        InitialVisBias = getBiasFromSamples(TrainData, 0.5)
    end
	return RBM(Float64, IsingSpin, IsingSpin, IsingActivation, MeansIsing, MargHiddenIsing, FlipIsing, n_vis, n_hid; InitVisBias=InitialVisBias[:,1])
end

function Base.show(io::IO, rbm::RBM{T,V,H}) where {T,V,H}
    n_vis = size(rbm.vbias, 1)
    n_hid = size(rbm.hbias, 1)
    print(io, "RBM{$V,$H}($n_vis, $n_hid)")
end

## utils

function PassVisToHid(rbm::RBM, vis::Mat{T}) where T
    p = rbm.W * vis .+ rbm.hbias
    return p
end

function PassHidToVis(rbm::RBM, hid::Mat{T}) where T
    p = rbm.W' * hid .+ rbm.vbias
    return p
end

function hid_prob_one(rbm::RBM, vis::Mat{T}) where T
    return rbm.activation(PassVisToHid(rbm,vis))
end

function vis_prob_one(rbm::RBM, hid::Mat{T}) where T
    return rbm.activation(PassHidToVis(rbm,hid))
end


## samping

function sample(::Type{Degenerate}, means::Mat{T}) where T
    return means
end

# prob : probability of being 1
function sample(::Type{Bernoulli}, prob::Mat{T}) where T
    return map(T, float((rand(size(prob)) .< prob)))
end

# prob : probability of being 1
function sample(::Type{IsingSpin}, prob::Mat{T}) where T
    return map(x -> x ? 1.0 : -1.0, rand(size(prob)) .< prob)
end

function sample(::Type{Gaussian}, means::Mat{T}) where T
    sigma2 = 1                   # using fixed standard diviation
    samples = zeros(T,size(means))
    for j=1:size(means, 2), i=1:size(means, 1)
        samples[i, j] = T(rand(Normal(means[i, j], sigma2)))
    end
    return samples
end


function sample_hiddens(rbm::AbstractRBM{T,V,H}, vis::Mat) where {T,V,H}
    prob_one = hid_prob_one(rbm, vis)
    return sample(H, prob_one), prob_one
end


function sample_visibles(rbm::AbstractRBM{T,V,H}, hid::Mat) where {T,V,H}
    prob_one = vis_prob_one(rbm, hid)
    return sample(V, prob_one), prob_one
end


function gibbs_training(rbm::AbstractRBM, vis::Mat; n_times=1)
    v_pos = vis
    h_sample, h_pos = sample_hiddens(rbm, v_pos)

    v_sample, v_prob = sample_visibles(rbm, h_sample)
    h_sample, h_prob = sample_hiddens(rbm, v_sample)
    for i=1:n_times-1
        v_sample, v_prog = sample_visibles(rbm, h_sample)
        h_sample, h_prob = sample_hiddens(rbm, v_sample)
    end
    v_neg = v_sample
    h_neg = h_prob

    h_pos = rbm.means(h_pos)
    h_neg = rbm.means(h_neg)

    return v_pos, h_pos, v_neg, h_neg
end


## scoring

function free_energy(rbm::RBM, vis::Mat)
    vb = sum(vis .* rbm.vbias, 1)

    fe_exp = rbm.marg(rbm.W * vis .+ rbm.hbias) # 1 + exp.(rbm.W * vis .+ rbm.hbias)
    tofinite!(fe_exp; nozeros=true)

    Wx_b_log = sum(log.(fe_exp), 1)
    result = - vb - Wx_b_log

    return result
end


function score_samples_tap(rbm::RBM, vis::Mat; n_iter=5)
    m_vis, m_hid = iter_mag(rbm, vis; approx="tap2", n_times=n_iter)

    # regularization
    eps=1e-6
    m_vis = max.(m_vis, -1+eps)
    m_hid = max.(m_hid, -1+eps)
    m_vis = min.(m_vis,1-eps)
    m_hid = min.(m_hid,1-eps)

    fe_tap = free_energy_tap(rbm, m_vis, m_hid)
    fe = free_energy(rbm,vis)

    return fe_tap - fe
end
"""
_, _, m_vis, m_hid = iter_mag(rbm, vis; n_times=n_iter, approx="tap2")
    eps=1e-6
    m_vis = max(m_vis, eps)
    m_vis = min(m_vis, 1.0-eps)
    m_hid = max(m_hid, eps)
    m_hid = min(m_hid, 1.0-eps)

    m_vis2 = abs2(m_vis)
    m_hid2 = abs2(m_hid)

    S = - sum(m_vis.*log(m_vis)+(1.0-m_vis).*log(1.0-m_vis),1) - sum(m_hid.*log(m_hid)+(1.0-m_hid).*log(1.0-m_hid),1)
    U_naive = - gemv('T',m_vis,rbm.vbias)' - gemv('T',m_hid,rbm.hbias)' - sum(gemm('N','N',rbm.W,m_vis).*m_hid,1)
    Onsager = - 0.5 * sum(gemm('N','N',rbm.W2,m_vis-m_vis2).*(m_hid-m_hid2),1)    
    fe_tap = U_naive + Onsager - S
    fe = free_energy(rbm, vis)
return fe_tap - fe
"""

function free_energy_tap(rbm::RBM, mag_vis::Mat, mag_hid::Mat) 
    mag_vis2 = abs2.(mag_vis)
    mag_hid2 = abs2.(mag_hid)


    Entropy = entropyMF(rbm, mag_vis, mag_hid)
    U_naive = -( sum(mag_vis.*rbm.vbias,1) + sum(mag_hid.*rbm.hbias,1) + sum(mag_hid.*(rbm.W*mag_vis),1) )
    Onsager = -( 0.5* sum( (rbm.W2*(1-mag_vis2)).*(1-mag_hid2),1 ) )
    fe_tap = U_naive+Onsager - Entropy
    return fe_tap
end



function entropyMF(rbm::RBM, mag_vis::Mat, mag_hid::Mat)
    S =  sum(entropy_bin(0.5*(1+mag_vis)) + entropy_bin(0.5*(1-mag_vis)),1)
    S += sum(entropy_bin(0.5*(1+mag_hid)) + entropy_bin(0.5*(1-mag_hid)),1)
    return S
end

function score_samples(rbm::AbstractRBM, vis::Mat;
                          sample_size=10000)
    if issparse(vis)
        # sparse matrices may be infeasible for this operation
        # so using only little sample
        cols = rand(1:size(vis, 2), sample_size)
        vis = full(vis[:, cols])
    end
    n_feat, n_samples = size(vis)
    vis_corrupted = copy(vis)
    idxs = rand(1:n_feat, n_samples)
    for (i, j) in zip(idxs, 1:n_samples)
        vis_corrupted[i, j] = rbm.flip(vis_corrupted[i, j]) # 1 - vis_corrupted[i, j]
    end

    fe = free_energy(rbm, vis)
    fe_corrupted = free_energy(rbm, vis_corrupted)
    fe_diff = fe_corrupted - fe
    tofinite!(fe_diff; nozeros=true)
    score_row =  n_feat * log.(logistic(fe_diff))

    result = map(Float64, squeeze(score_row', 2))
    tofinite!(result)

    return result
end

function pseudo_likelihood(rbm::AbstractRBM, X)
    N=size(X,1)
    return mean(score_samples(rbm, X))/N
end


function pseudo_likelihood_tap(rbm::AbstractRBM, X)
    N=size(X,1)
    mean(score_samples_tap(rbm, X))/N
end

## gradient calculation

"""
Contrastive divergence sampler. Options:

 * n_gibbs - number of gibbs sampling loops (default: 1)
"""
function contdiv(rbm::AbstractRBM, vis::Mat, ctx::Dict)
    n_gibbs = @get(ctx, :n_gibbs, 1)
    v_pos, h_pos, v_neg, h_neg = gibbs_training(rbm, vis, n_times=n_gibbs)
    return v_pos, h_pos, v_neg, h_neg
end

"""
Persistent contrastive divergence sampler. Options:

 * n_gibbs - number of gibbs sampling loops
"""
function persistent_contdiv(rbm::AbstractRBM, vis::Mat, ctx::Dict)
    n_gibbs = @get(ctx, :n_gibbs, 1)
    persistent_chain = @get_array(ctx, :persistent_chain, size(vis), vis)
    if size(persistent_chain) != size(vis)
        println("persistent_chain not initialized")
        # persistent_chain not initialized or batch size changed
        # re-initialize
        persistent_chain = vis
    end
    # take positive samples from real data (get h_pos)
    v_pos, h_pos, _, _ = gibbs_training(rbm, vis)
    # take negative samples from "fantasy particles"
    _, _, v_neg, h_neg = gibbs_training(rbm, persistent_chain, n_times=n_gibbs)
    copy!(ctx[:persistent_chain], v_neg)
    return v_pos, h_pos, v_neg, h_neg
end

"""
Function for calculating gradient of negative log-likelihood of the data.
Options:

 * :sampler - sampler to use (default: persistent_contdiv)

Returns:

 * (dW, db, dc) - tuple of gradients for weights, visible and hidden biases,
                  respectively
"""
function gradient_classic(rbm::RBM, vis::Mat{T}, ctx::Dict) where T
    sampler = @get_or_create(ctx, :sampler, persistent_contdiv)
    v_pos, h_pos, v_neg, h_neg = sampler(rbm, vis, ctx)
    dW = @get_array(ctx, :dW_buf, size(rbm.W), similar(rbm.W))
    n_obs = size(vis, 2)
    # same as: dW = ((h_pos * v_pos') - (h_neg * v_neg')) / n_obs
    gemm!('N', 'T', T(1 / n_obs), h_neg, v_neg, T(0.0), dW)
    gemm!('N', 'T', T(1 / n_obs), h_pos, v_pos, T(-1.0), dW)
    # gradient for vbias and hbias
    db = squeeze(sum(v_pos, 2) - sum(v_neg, 2), 2) ./ n_obs
    dc = squeeze(sum(h_pos, 2) - sum(h_neg, 2), 2) ./ n_obs
    return dW, db, dc
end


## updating

function grad_apply_learning_rate!(rbm::RBM{T,V,H}, X::Mat,
                                   dtheta::Tuple, ctx::Dict) where {T,V,H}
    dW, db, dc = dtheta
    lr = T(@get(ctx, :lr, 0.1))
    # same as: dW *= lr
    scal!(length(dW), lr, dW, 1)
    scal!(length(db), lr, db, 1)
    scal!(length(dc), lr, dc, 1)
end


function grad_apply_momentum!(rbm::RBM{T}, X::Mat,
                              dtheta::Tuple, ctx::Dict) where T
    dW, db, dc = dtheta
    momentum = @get(ctx, :momentum, 0.9)
    dW_prev = @get_array(ctx, :dW_prev, size(dW), zeros(T, size(dW)))
    # same as: dW += momentum * dW_prev
    axpy!(momentum, dW_prev, dW)
end


function grad_apply_weight_decay!(rbm::RBM, X::Mat,
                                  dtheta::Tuple, ctx::Dict)
    # The decay penalty should drive all weights toward
    # zero by some small amount on each update.
    dW, db, dc = dtheta
    decay_kind = @get_or_return(ctx, :weight_decay_kind, nothing)
    decay_rate = @get(ctx, :weight_decay_rate,
                      throw(ArgumentError("If using :weight_decay_kind, weight_decay_rate should also be specified")))
    is_l2 = @get(ctx, :l2, false)
    if decay_kind == :l2
        # same as: dW -= decay_rate * W
        axpy!(-decay_rate, rbm.W, dW)
    elseif decay_kind == :l1
        # same as: dW -= decay_rate * sign(W)
        axpy!(-decay_rate, sign(rbm.W), dW)
    end

end

function grad_apply_sparsity!(rbm::RBM{T}, X::Mat,
                           dtheta::Tuple, ctx::Dict) where T
    # The sparsity constraint should only drive the weights
    # down when the mean activation of hidden units is higher
    # than the expected (hence why it isn't squared or the abs())
    dW, db, dc = dtheta
    cost = @get_or_return(ctx, :sparsity_cost, nothing)
    target = @get(ctx, :sparsity_target, throw(ArgumentError("If :sparsity_cost is used, :sparsity_target should also be defined")))
    curr_sparsity = mean(hid_means(rbm, X))
    penalty = T(cost * (curr_sparsity - target))
    add!(dW, -penalty)
    add!(db, -penalty)
    add!(dc, -penalty)
end


function update_weights!(rbm::RBM, dtheta::Tuple, ctx::Dict)
    dW, db, dc = dtheta
    axpy!(1.0, dW, rbm.W)
    rbm.vbias += db
    rbm.hbias += dc
    # save previous dW
    dW_prev = @get_array(ctx, :dW_prev, size(dW), similar(dW))
    copy!(dW_prev, dW)
end


"""
Update RBM parameters using provided tuple `dtheta = (dW, db, dc)` of
parameter gradients. Before updating weights, following transformations
are applied to gradients:

 * learning rate (see `grad_apply_learning_rate!` for details)
 * momentum (see `grad_apply_momentum!` for details)
 * weight decay (see `grad_apply_weight_decay!` for details)
 * sparsity (see `grad_apply_sparsity!` for details)
"""
function update_classic!(rbm::RBM, X::Mat, dtheta::Tuple, ctx::Dict)
    # apply gradient updaters. note, that updaters all have
    # the same signature and are thus composable
    grad_apply_learning_rate!(rbm, X, dtheta, ctx)
    grad_apply_momentum!(rbm, X, dtheta, ctx)
    grad_apply_weight_decay!(rbm, X, dtheta, ctx)
    grad_apply_sparsity!(rbm, X, dtheta, ctx)
    # add gradient to the weight matrix
    update_weights!(rbm, dtheta, ctx)
end


## fitting

function fit_batch!(rbm::RBM, X::Mat, ctx = Dict())
    grad = @get_or_create(ctx, :gradient, gradient_classic)
    upd = @get_or_create(ctx, :update, update_classic!)
    dtheta = grad(rbm, X, ctx)
    upd(rbm, X, dtheta, ctx)
    return rbm
end


"""
Fit RBM to data `X`. Options that can be provided in the `opts` dictionary:

 * :n_epochs - number of full loops over data (default: 10)
 * :batch_size - size of mini-batches to use (default: 100)
 * :randomize - boolean, whether to shuffle batches or not (default: false)
 * :gradient - function to use for calculating parameter gradients
               (default: gradient_classic)
 * :update - function to use to update weights using calculated gradient
             (default: update_classic!)
 * :scorer - function to calculate how good the model is at the moment
             (default: pseudo_likelihood)
 * :reporter - type for reporting intermediate results using `report()`
               function (default: TextReporter)

Each function can additionally take other options, see their
docstrings/code for details.

NOTE: this function is incremental, so one can, for example, run it for
10 epochs, then inspect the model, then run it for 10 more epochs
and check the difference.
"""
function fit(rbm::RBM{T}, X::Mat, opts::Dict{Any,Any}) where T
    #@assert minimum(X) >= 0 && maximum(X) <= 1
    ctx = copy(opts)
    check_options(ctx)
    n_examples = size(X, 2)
    batch_size = @get(ctx, :batch_size, 100)
    n_batches = Int(ceil(n_examples/batch_size))
    batch_idxs = split_evenly(n_examples, batch_size)
    if @get(ctx, :randomize, false)
        batch_idxs = sample(batch_idxs, length(batch_idxs); replace=false)
    end
    n_epochs = @get(ctx, :n_epochs, 10)
    # scorer = @get_or_create(ctx, :scorer, pseudo_likelihood)    
    scorer = @get_or_create(ctx, :scorer, pseudo_likelihood_tap)
    reporter = @get_or_create(ctx, :reporter, TextReporter)
    println("INIT ",scorer(rbm, X))
    for epoch=1:n_epochs
        epoch_time = @elapsed begin
            current_batch = 0
            for (batch_start, batch_end) in batch_idxs
                # BLAS.gemm! can't handle sparse matrices, so cheaper
                # to make it dense here
                batch = full(X[:, batch_start:batch_end])
                batch = ensure_type(T, batch)
                current_batch += 1
                fit_batch!(rbm, batch, ctx)
                #println(current_batch)
                #println((typeof(reporter) <: BatchReporter) && (current_batch % reporter.every) == 0)
                if ((typeof(reporter) <: BatchReporter) &&
                    (current_batch % reporter.every) == 0)
		    report(reporter, rbm, epoch, current_batch, scorer, X, ctx)
		end
            end
        end
        #score = scorer(rbm, X)
        if typeof(reporter) <: EpochReporter 
          report(reporter, rbm, epoch, epoch_time, scorer, X, ctx)
        end
    end
    return rbm
end

fit(rbm::RBM, X::Mat; opts...) = fit(rbm, X, Dict{Any,Any}(opts))


## operations on learned RBM

"""Base data X through trained RBM to obtain compressed representation"""
function transform(rbm::RBM{T}, X::Mat) where T
    return hid_means(rbm, ensure_type(T, X))
end


"""Given trained RBM and sample of visible data, generate similar items"""
function generate(rbm::RBM{T}, vis::Vec; n_gibbs=1) where T
    return gibbs_training(rbm, reshape(ensure_type(T, vis), length(vis), 1); n_times=n_gibbs)[3]
end

function generate(rbm::RBM{T}, X::Mat; n_gibbs=1) where T
    return gibbs_training(rbm, ensure_type(T, X); n_times=n_gibbs)[3]
end


"""
Get weight matrix of a trained RBM. Options:

 * transpose - boolean, whether to transpose weight matrix (convenient)
               or not (efficient). Default: true
"""
function coef(rbm::RBM; transpose=true)
    return if transpose rbm.W' else rbm.W end
end
# synonyms
weights = coef

"""Get biases of hidden units"""
hbias(rbm::RBM) = rbm.hbias

"""Get biases of visible units"""
vbias(rbm::RBM) = rbm.vbias


function mag_vis_nmf(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})x&
    return(tanh.(PassHidToVis(rbm,m_hid)))
end

function mag_hid_nmf(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})
    return(tanh.(PassVisToHid(rbm,m_vis)))
end

function mag_vis_tap2(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})
    # See paper of federico
    tanh.(PassHidToVis(rbm,m_hid) - (rbm.W2'*(1-abs2.(m_hid))).*m_vis)
end

function mag_hid_tap2(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})
    # See paper of federico
    tanh.(PassVisToHid(rbm,m_vis) - (rbm.W2*(1-abs2.(m_vis))).*m_hid)
end

function iter_mag_training(rbm::RBM, vis::Mat{Float64}, ctx::Dict; n_times=1)
    approx = @get(ctx, :approx, "tap2")
    dump =   @get(ctx, :dump ,0.5)

    v_pos = vis
    h_pos = mag_hid_nmf(rbm,vis,zeros(1,1)) #rbm.activation(PassVisToHid(rbm,vis))
    if approx == "nmf"
        mag_vis = mag_vis_nmf
        mag_hid = mag_hid_nmf
    elseif approx == "tap2"
        mag_vis = mag_vis_tap2
        mag_hid = mag_hid_tap2
    end

    m_vis = (1-dump) * mag_vis(rbm, vis, h_pos) + dump * vis
    m_hid = (1-dump) * mag_hid(rbm, m_vis, h_pos) + dump * h_pos
    for i=1:n_times-1
       m_vis = (1-dump) * mag_vis(rbm, m_vis, m_hid) + dump * m_vis
       m_hid = (1-dump) * mag_hid(rbm, m_vis, m_hid) + dump * m_hid
    end

    return v_pos, h_pos, m_vis, m_hid
end

function iter_mag(rbm::RBM, vis::Mat{Float64}; approx="tap2", n_times=1, dump=0.5)
    #approx = @get(ctx, :approx, "tap2")
    #dump =   @get(ctx, :dump ,0.5)

    v_pos = vis
    h_pos = mag_hid_nmf(rbm,vis,zeros(1,1))

    if approx == "nmf"
        mag_vis = mag_vis_nmf
        mag_hid = mag_hid_nmf
    elseif approx == "tap2"
        mag_vis = mag_vis_tap2
        mag_hid = mag_hid_tap2
    end

    m_vis = (1-dump) * mag_vis(rbm, vis, h_pos) + dump * vis
    m_hid = (1-dump) * mag_hid(rbm, m_vis, h_pos) + dump * h_pos
    for i=1:n_times-1
       m_vis = (1-dump) * mag_vis(rbm, m_vis, m_hid) + dump * m_vis
       m_hid = (1-dump) * mag_hid(rbm, m_vis, m_hid) + dump * m_hid
    end
    return m_vis, m_hid
end


function sampler_UpdMeanfield(rbm::RBM, vis::Mat, ctx::Dict)
    n_gibbs = @get(ctx, :n_gibbs, 1)
    v_pos, h_pos, v_neg, h_neg = iter_mag_training(rbm, vis, ctx; n_times=n_gibbs)
    #println(v_neg)
    #readline(STDIN)
    return v_pos, h_pos, v_neg, h_neg
end


function persistent_meanfield(rbm::RBM, vis::Mat, ctx::Dict)
    rbm.W2 = abs2.(rbm.W)
    n_gibbs = @get(ctx, :n_gibbs, 1)
    persistent_chain = @get_array(ctx, :persistent_chain, size(vis), vis)
    if size(persistent_chain) != size(vis)
        println("persistent_chain not initialized")
        # persistent_chain not initialized or batch size changed
        # re-initialize
        persistent_chain = vis
    end
    # take positive samples from real data (get h_pos)
    v_pos, h_pos, _, _ = iter_mag_training(rbm, vis, ctx)
    # take negative samples from "fantasy particles"
    _, _, v_neg, h_neg = iter_mag_training(rbm, persistent_chain, ctx; n_times=n_gibbs)
    copy!(ctx[:persistent_chain], v_neg)
    return v_pos, h_pos, v_neg, h_neg
end

