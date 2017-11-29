
abstract type AbstractReporter end

abstract type EpochReporter <: AbstractReporter end
abstract type BatchReporter <: AbstractReporter end

mutable struct TextReporter <: EpochReporter
 #   exec::Function
end

#function TextReporter(rep::Function) 
    #TextReporter(rep)
#end

function report(epoch::Int, epoch_time::Float64, score::Float64, ctx::Dict{Any,Any})
    println("[Epoch $epoch] Score: $score [$(epoch_time)s]")
end

#function report(r::TextReporter, dbn::DBN, epoch::Int, layer::Int, ctx::Dict{Any,Any})
#    println("[Layer $layer] Starting epoch $epoch")
#end

