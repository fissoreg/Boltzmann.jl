
abstract type AbstractReporter end

abstract type EpochReporter <: AbstractReporter end
abstract type BatchReporter <: AbstractReporter end

struct TextReporter <: EpochReporter end

function report(r::TextReporter, rbm::AbstractRBM,
                epoch::Int, epoch_time::Float64, scorer::Function, X::Mat, ctx::Dict{Any,Any})
    println("[Epoch $epoch] Score: $(scorer(rbm, X)) [$(epoch_time)s]")
end

#function report(r::TextReporter, dbn::DBN, epoch::Int, layer::Int, ctx::Dict{Any,Any})
#    println("[Layer $layer] Starting epoch $epoch")
#end

