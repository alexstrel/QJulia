module QJuliaFieldsBlas

import QJuliaRegisters
import QJuliaEnums
import QJuliaBlas

#we need to overload basic operators
#import Base.+
#import Base.*
#import Base.-
#import Base.=
#import Base./

#Currently only Complex{T} is supported

@inline function xpy(x::QJuliaGenericField_qj, y::QJuliaGenericField_qj); y.v .=@. x.v + y.v; end

@inline function xmy(x::QJuliaGenericField_qj, y::QJuliaGenericField_qj); y.v .=@. x.v - y.v; end 

@inline function axpy(a::T, x::QJuliaGenericField_qj, y::QJuliaGenericField_qj) where T <: AbstractFloat
                 #this will reinterpret matrix as long vectors
                 rxv = view(reinterpret(x.field_desc.prec, x.v), :) 
                 ryv = view(reinterpret(y.field_desc.prec, y.v), :)
                 ryv .=@. a*rxv + ryv
end #axpy

@inline function axpy_m(a::T, x::QJuliaGenericField_qj, y::QJuliaGenericField_qj) where T <: AbstractFloat
                 rxv = view(reinterpret(x.field_desc.prec, x.v), :, :) 
                 ryv = view(reinterpret(y.field_desc.prec, y.v), :, :)
                 ryv .=@. a*rxv + ryv
end #axpy_m

@inline function xpay(x::QJuliaGenericField_qj, a::T, y::QJuliaGenericField_qj) where T <: AbstractFloat
                 #this will reinterpret matrix as long vectors
                 rxv = view(reinterpret(x.field_desc.prec, x.v), :) 
                 ryv = view(reinterpret(y.field_desc.prec, y.v), :)
                 ryv .=@. rxv + a*ryv
end #xpay

@inline function xpay_m(x::QJuliaGenericField_qj, a::T, y::QJuliaGenericField_qj) where T <: AbstractFloat
                 rxv = view(reinterpret(x.field_desc.prec, x.v), :, :) 
                 ryv = view(reinterpret(y.field_desc.prec, y.v), :, :)
                 ryv .=@. rxv + a*ryv
end #xpay_m

@inline function caxpy(a::Complex{T}, x::QJuliaGenericField_qj, y::QJuliaGenericField_qj) where T <: AbstractFloat
                 y.v .=@. a*x.v + y.v
end #caxpy

@inline function cxpay(x::QJuliaGenericField_qj, a::Complex{T}, y::QJuliaGenericField_qj) where T <: AbstractFloat
                 y.v .=@. x.v + a*y.v
end #cxpay

@inline function caxpyXmaz(a::Complex{T}, x::QJuliaGenericField_qj, y::QJuliaGenericField_qj, z::QJuliaGenericField_qj)  where T <: AbstractFloat 

@threads for i in 1:length(x.v)
           y.v[i]  += a*x.v[i]
           x.v[i]  -= a*z.v[i]
         end
end #caxpyXmaz

#First performs the operation x[i] = x[i] + a*p[i]
#Second performs the operator p[i] = u[i] + b*p[i]

@inline function axpyZpbx(a::T, p::QJuliaGenericField_qj, x::QJuliaGenericField_qj, u::QJuliaGenericField_qj, b::T)  where T <: AbstractFloat 

         #this will reinterpret matrix as long vectors
         rpv = view(reinterpret(p.field_desc.prec, p.v), :) 
         rxv = view(reinterpret(x.field_desc.prec, x.v), :)
         ruv = view(reinterpret(u.field_desc.prec, u.v), :)

@threads for i in 1:length(rxv)
           rxv[i] = rxv[i]+a*rpv[i] 
           rpv[i] = ruv[i]+b*rpv[i] 
         end
end #axpyZpbx

@inline function axpyZpbx_m(a::T, p::QJuliaGenericField_qj, x::QJuliaGenericField_qj, u::QJuliaGenericField_qj, b::T)  where T <: AbstractFloat 

         #this will reinterpret matrix as long vectors
         rpv = view(reinterpret(p.field_desc.prec, p.v), :, :) 
         rxv = view(reinterpret(x.field_desc.prec, x.v), :, :)
         ruv = view(reinterpret(u.field_desc.prec, u.v), :, :)

@threads for i in 1:length(rxv)
           rxv[i] = rxv[i]+a*rpv[i] 
           rpv[i] = ruv[i]+b*rpv[i] 
         end
end #axpyZpbx_m

end #QJuliaFieldsBlas


