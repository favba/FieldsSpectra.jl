module FieldsSpectra
export compute_shells, compute_shells2D, calculate_u1u2_spectrum, calculate_vector_spectrum, calculate_scalar_spectrum, squared_mean, proj_mean
export power_spectrum, power_spectrum!, hpower_spectrum, hpower_spectrum!

using FluidFields

@inline proj(a::Complex,b::Complex) = muladd(a.re, b.re, a.im*b.im)

function compute_shells(kx::AbstractVector{T},ky::AbstractVector,kz::AbstractVector) where {T}
    Nx = length(kx)
    Ny = length(ky)
    Nz = length(kz)
    nShells = min(Nx,Ny÷2,Nz÷2)
    maxdk = max(kx[2],ky[2],kz[2])
    kh = zeros(T,nShells)
    numPtsInShell = zeros(Int,nShells)

    @inbounds for k in 1:Nz
        kz2 = kz[k]^2
        (round(Int,sqrt(kz2)/maxdk)+1) > nShells && break
        for j=1:Ny
            kzy2 = kz2 + ky[j]^2
            (round(Int,sqrt(kzy2)/maxdk)+1) > nShells && break
            for i=1:Nx
                K = sqrt(muladd(kx[i],kx[i],kzy2))
                ii = round(Int,K/maxdk)+1
                ii > nShells && break
                kh[ii] += K
                numPtsInShell[ii] += 1
            end
        end
    end
  
    @inbounds @simd for i in 1:length(kh)
        kh[i] = kh[i]/numPtsInShell[i]
    end

    return kh
end

compute_shells(f::AbstractField) = compute_shells(f.kx,f.ky,f.kz)

function compute_shells2D(kx::AbstractVector{T},ky) where {T}
    Nx = length(kx)
    Ny = length(ky)
    nShells2D = min(Nx,Ny÷2)
    maxdk2D = max(kx[2],ky[2])
    kh = zeros(T,nShells2D)
    numPtsInShell2D = zeros(Int,nShells2D)

    @inbounds for j=1:Ny
        ky2 = ky[j]^2
        (round(Int,sqrt(ky2)/maxdk2D)+1) > nShells2D && break
        for i=1:Nx
            K = sqrt(muladd(kx[i],kx[i], ky2))
            ii = round(Int,K/maxdk2D)+1
            ii > nShells2D && break
            kh[ii] += K
            numPtsInShell2D[ii] += 1
        end
    end
  
    @inbounds @simd for i in 1:length(kh)
        kh[i] = kh[i]/numPtsInShell2D[i]
    end

    return kh
end

compute_shells2D(f::AbstractField) = compute_shells2D(f.kx,f.ky)

function calculate_u1u2_spectrum!(Ef,u::VectorField{T},cplane::Int=1) where {T}
    isrealspace(u) && fourier!(u)
    ux = u.c.x
    uy = u.c.y
    KX = u.kx
    KY = u.ky
    NX = size(u,1)
    NY = size(u,2)
    # Initialize the shells to zeros
    fill!(Ef,zero(T))
    maxdk2d = max(KX[2],KY[2])
    nshells = min(NX,NY÷2)
    @inbounds for j in 1:NY
        KY2 = KY[j]^2
        n = round(Int, sqrt(KY2)/maxdk2d) + 1
        n > nshells && break
        magsq = abs2(ux[1,j,cplane]) + abs2(uy[1,j,cplane])
        ee = magsq
        Ef[n]+=ee
        for i in 2:NX
            k = sqrt(muladd(KX[i],KX[i], KY2))
            n = round(Int, k/maxdk2d) + 1
            n > nshells && break
            magsq = abs2(ux[i,j,cplane]) + abs2(uy[i,j,cplane])
            ee = 2*magsq
            Ef[n]+=ee
        end
    end
    return Ef
end

function calculate_u1u2_spectrum(u::VectorField{T},p::Int=1) where {T}
    NX = size(u,1)
    NY = size(u,2)
    Ef = zeros(T,min(NX,NY÷2))
    return calculate_u1u2_spectrum!(Ef,u,p) 
end

function calculate_vector_spectrum!(Ef,u::VectorField{T}) where {T}
    isrealspace(u) && fourier!(u)
    ux = u.c.x
    uy = u.c.y
    uz = u.c.z
    KX = u.kx
    KY = u.ky
    KZ = u.kz
    NX = size(u,1)
    NY = size(u,2)
    NZ = size(u,3)
    # Initialize the shells to zeros
    fill!(Ef,zero(T))
    maxdk2d = max(KX[2],KY[2],KZ[2])
    vK = KX[2]*KY[2]*KZ[2]
    nshells = min(NX,NY÷2,NZ÷2)

    fix = oneunit(T)
    @inbounds for l in 1:NZ
        KZ2 = KZ[l]^2
        (round(Int, sqrt(KZ2)/maxdk2d) + 1) > nshells && break
        @inbounds for j in 1:NY
            KY2 = KY[j]^2 + KZ2
            K = sqrt(KY2)
            n = round(Int, K/maxdk2d) + 1
            n > nshells && break
            magsq = abs2(ux[1,j,l]) + abs2(uy[1,j,l]) + abs2(uz[1,j,l])
            ee = magsq
            Ef[n]+=ee
            fix = zero(T)
            for i in 2:NX
                K = sqrt(muladd(KX[i],KX[i], KY2))
                n = round(Int, K/maxdk2d) + 1
                n > nshells && break
                magsq = abs2(ux[i,j,l]) + abs2(uy[i,j,l]) + abs2(uz[i,j,l])
                ee = 2*magsq
                Ef[n]+=ee
            end
        end
    end
    return Ef
end

function calculate_vector_spectrum(u::VectorField{T}) where {T}
    NX = size(u,1)
    NY = size(u,2)
    NZ = size(u,3)
    Ef = zeros(T,min(NX,NY÷2,NZ÷2))
    return calculate_vector_spectrum!(Ef,u) 
end

function calculate_scalar_spectrum!(Ef,u::ScalarField{T}) where {T}
    isrealspace(u) && fourier!(u)
    KX = u.kx
    KY = u.ky
    KZ = u.kz
    NX = size(u,1)
    NY = size(u,2)
    NZ = size(u,3)
    # Initialize the shells to zeros
    fill!(Ef,zero(T))
    maxdk2d = max(KX[2],KY[2],KZ[2])
    nshells = min(NX,NY÷2,NZ÷2)

    @inbounds for l in 1:NZ
        KZ2 = KZ[l]^2
        (round(Int, sqrt(KZ2)/maxdk2d) + 1) > nshells && break
        @inbounds for j in 1:NY
            KY2 = KY[j]^2 + KZ2
            n = round(Int, sqrt(KY2)/maxdk2d) + 1
            n > nshells && break
            magsq = abs2(u[1,j,l])
            ee = magsq
            Ef[n]+=ee
            for i in 2:NX
                k = sqrt(muladd(KX[i],KX[i], KY2))
                n = round(Int, k/maxdk2d) + 1
                n > nshells && break
                magsq = abs2(u[i,j,l])
                ee = 2*magsq
                Ef[n]+=ee
            end
        end
    end
    return Ef
end

function calculate_scalar_spectrum(u::ScalarField{T}) where {T}
    NX = size(u,1)
    NY = size(u,2)
    NZ = size(u,3)
    Ef = zeros(T,min(NX,NY÷2,NZ÷2))
    return calculate_scalar_spectrum!(Ef,u) 
end

function squared_mean(u::ScalarField{T}) where {T}
    isrealspace(u) && fourier!(u)
    ee = zero(T)
    @inbounds for l in axes(u,3)
        @inbounds for j in axes(u,2)
            @simd for i in axes(u,1)
                magsq = abs2(u[i,j,l])
                ee += (1 + (i>1))*magsq 
            end
        end
    end
    return ee
end

function proj_mean(u::ScalarField{T},v::ScalarField{T2}) where {T,T2}
    isrealspace(u) && fourier!(u)
    isrealspace(v) && fourier!(v)
    ee = zero(promote_type(T,T2))
    @inbounds for l in axes(u,3)
        @inbounds for j in axes(u,2)
            @simd for i in axes(u,1)
                magsq = proj(u[i,j,l],v[i,j,l])
                ee += (1 + (i>1))*magsq 
            end
        end
    end
    return ee
end

function power_spectrum!(Ef::Vector{T},u::ScalarField{T},v::ScalarField{T}=u) where {T}
    isrealspace(u) && fourier!(u)
    isrealspace(v) && fourier!(v)
    KX = u.kx
    KY = u.ky
    KZ = u.kz
    NX = size(u,1)
    NY = size(u,2)
    NZ = size(u,3)
    # Initialize the shells to zeros
    fill!(Ef,zero(T))
    maxdk2d = max(KX[2],KY[2],KZ[2])
    nshells = min(NX,NY÷2,NZ÷2)

    @inbounds for l in 1:NZ
        KZ2 = KZ[l]^2
        (round(Int, sqrt(KZ2)/maxdk2d) + 1) > nshells && break
        @inbounds for j in 1:NY
            KY2 = KY[j]^2 + KZ2
            n = round(Int, sqrt(KY2)/maxdk2d) + 1
            n > nshells && break
            magsq = proj(u[1,j,l],v[1,j,l])
            ee = magsq
            Ef[n]+=ee
            for i in 2:NX
                k = sqrt(muladd(KX[i],KX[i], KY2))
                n = round(Int, k/maxdk2d) + 1
                n > nshells && break
                magsq = proj(u[i,j,l],v[i,j,l])
                ee = 2*magsq
                Ef[n]+=ee
            end
        end
    end
    return Ef
end

function power_spectrum(u::ScalarField{T},v::ScalarField=u) where {T}
    NX = size(u,1)
    NY = size(u,2)
    NZ = size(u,3)
    Ef = zeros(T,min(NX,NY÷2,NZ÷2))
    return power_spectrum!(Ef,u,v) 
end

function hpower_spectrum!(Ef::Vector{T},u::ScalarField{T},v::ScalarField{T}=u,p::Integer=1) where {T}
    isrealspace(u) && fourier!(u)
    isrealspace(v) && fourier!(v)
    KX = u.kx
    KY = u.ky
    NX = size(u,1)
    NY = size(u,2)
    # Initialize the shells to zeros
    fill!(Ef,zero(T))
    maxdk2d = max(KX[2],KY[2])
    nshells = min(NX,NY÷2)

    @inbounds for j in 1:NY
        KY2 = KY[j]^2
        n = round(Int, sqrt(KY2)/maxdk2d) + 1
        n > nshells && break
        magsq = proj(u[1,j,p],v[1,j,p])
        ee = magsq
        Ef[n]+=ee
        for i in 2:NX
            k = sqrt(muladd(KX[i],KX[i], KY2))
            n = round(Int, k/maxdk2d) + 1
            n > nshells && break
            magsq = proj(u[i,j,p],v[i,j,p])
            ee = 2*magsq
            Ef[n]+=ee
        end
    end
    return Ef
end

hpower_spectrum!(Ef::Vector{T},u::ScalarField{T},p::Integer) where {T} = hpower_spectrum!(Ef,u,u,p)

function hpower_spectrum(u::ScalarField{T},v::ScalarField=u,p::Integer=1) where {T}
    NX = size(u,1)
    NY = size(u,2)
    Ef = zeros(T,min(NX,NY÷2))
    return hpower_spectrum!(Ef,u,v,p) 
end

hpower_spectrum(u::ScalarField,p::Integer) = hpower_spectrum(u,u,p)

end # module
