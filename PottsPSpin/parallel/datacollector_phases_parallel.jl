using Pkg
const SCRIPT_DIR = @__DIR__
Pkg.activate(joinpath(SCRIPT_DIR, "..", ".."))

using JLD2, IterTools, LaTeXStrings
using DataFrames
using Random
using GraphicalModelLearning
using Base.Threads

include(joinpath(SCRIPT_DIR, "..", "..", "samplers.jl"))
include(joinpath(SCRIPT_DIR, "..", "..", "utils.jl"))
include(joinpath(SCRIPT_DIR, "..", "potts_tensors.jl"))
include(joinpath(SCRIPT_DIR, "..", "potts_samplers.jl"))

const OUTPUT_DIR = joinpath(SCRIPT_DIR, "thread_outputs")
const MERGED_FILE = joinpath(SCRIPT_DIR, "data_energies_high_parallel.jld2")

# Resume behavior:
# - true: skip jobs already present in per-thread shard files (by job_id)
# - false: run all jobs again and append to shards
resume_skip_completed = true

repeats = 3

nspins_list = [50]
β_list = range(0.3, 1.4, 8)
sampler_list = [:GlauberMin, :GlauberRand]
M_list = [10^4]
model_list = [:Ferro1, :Ferro2]
q_list = [2, 3, 4]

function empty_results_df()
    DataFrame(
        job_id = Int[],
        n_spins = Int[],
        β = Float64[],
        q = Int[],
        active_edges = Any[],
        sampler = Symbol[],
        model_type = Symbol[],
        n_samples = Int[],
        sample_energy = Float64[],
    )
end

thread_shard_path(tid::Int) = joinpath(OUTPUT_DIR, "thread_$(tid).jld2")

function load_df_or_empty(path::AbstractString)
    if !isfile(path)
        return empty_results_df()
    end
    try
        return jldopen(path, "r") do file
            file["df"]
        end
    catch err
        @warn "Failed to read shard; treating as empty" path err
        return empty_results_df()
    end
end

function append_row_to_shard!(path::AbstractString, row::NamedTuple)
    mkpath(dirname(path))
    if isfile(path)
        jldopen(path, "r+") do file
            df = file["df"]
            push!(df, row)
            if haskey(file, "df")
                delete!(file, "df")
            end
            file["df"] = df
        end
    else
        df = empty_results_df()
        push!(df, row)
        jldopen(path, "w") do file
            file["df"] = df
        end
    end
    return nothing
end

function collect_completed_job_ids(dir::AbstractString)
    completed = Set{Int}()
    if !isdir(dir)
        return completed
    end

    shard_paths = sort(filter(p -> endswith(p, ".jld2"), readdir(dir; join = true)))
    for path in shard_paths
        df = load_df_or_empty(path)
        if :job_id in names(df)
            union!(completed, Int.(df.job_id))
        end
    end
    return completed
end

function merge_shards!(dir::AbstractString, merged_file::AbstractString)
    if !isdir(dir)
        merged_df = empty_results_df()
        jldopen(merged_file, "w") do file
            file["df"] = merged_df
        end
        return merged_df
    end

    shard_paths = sort(filter(p -> endswith(p, ".jld2"), readdir(dir; join = true)))
    merged_df = empty_results_df()

    for path in shard_paths
        df = load_df_or_empty(path)
        nrow(df) == 0 && continue
        merged_df = nrow(merged_df) == 0 ? df : vcat(merged_df, df; cols = :union)
    end

    if nrow(merged_df) > 0 && (:job_id in names(merged_df))
        sort!(merged_df, :job_id)
        unique!(merged_df, :job_id)
    end

    jldopen(merged_file, "w") do file
        file["df"] = merged_df
    end
    return merged_df
end

base_arguments = collect(product(nspins_list, β_list, q_list, sampler_list, model_list, M_list))
jobs = collect(enumerate([arg for arg in base_arguments for _ in 1:repeats]))
total_jobs = length(jobs)

completed_job_ids = resume_skip_completed ? collect_completed_job_ids(OUTPUT_DIR) : Set{Int}()
pending_jobs = [job for job in jobs if !(job[1] in completed_job_ids)]

@info "Parallel run setup" threads = nthreads() total_jobs total_completed = length(completed_job_ids) pending = length(pending_jobs) output_dir = OUTPUT_DIR

done_counter = Atomic{Int}(0)
pending_total = length(pending_jobs)

@threads for j in 1:pending_total
    job_id, arg = pending_jobs[j]
    (ns, β, q, sampler_id, model_type, n_samples) = arg

    @assert iseven(ns)

    n_edges = div(3 * ns, 2)
    Edges = random_p_hyergraph(ns, n_edges, p = 3)
    while length(Edges) != n_edges
        Edges = random_p_hyergraph(ns, n_edges, p = 3)
    end

    Φ = if model_type == :Ferro1
        threealphabet_ferro1(q)
    elseif model_type == :Ferro2
        threealphabet_ferro2(q)
    else
        error("Unknown model_type: $model_type")
    end

    Energy(σ) = HypergraphPottsEnergy(σ, Edges, Φ, -1)

    energy_estim = if sampler_id == :GlauberMin
        Glauber_sampler(
            Energy,
            n_samples,
            ns,
            q;
            order_parameters = [Energy],
            return_samples = false,
            init = () -> Int.(ones(ns)),
            β = β,
            restarts = 100,
        )
    elseif sampler_id == :GlauberRand
        Glauber_sampler(
            Energy,
            n_samples,
            ns,
            q;
            order_parameters = [Energy],
            return_samples = false,
            β = β,
            restarts = 100,
        )
    else
        error("Unknown sampler_id: $sampler_id")
    end

    row = (
        job_id = job_id,
        n_spins = ns,
        β = β,
        q = q,
        active_edges = Edges,
        sampler = sampler_id,
        model_type = model_type,
        n_samples = n_samples,
        sample_energy = energy_estim[1],
    )

    append_row_to_shard!(thread_shard_path(threadid()), row)

    done_now = atomic_add!(done_counter, 1) + 1
    if done_now == 1 || done_now % 10 == 0 || done_now == pending_total
        @info "Progress" done = done_now pending_total remaining = pending_total - done_now
    end
end

merged_df = merge_shards!(OUTPUT_DIR, MERGED_FILE)
@info "Done" merged_file = MERGED_FILE rows = nrow(merged_df)
