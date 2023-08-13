import  casadi as ad 


x = ad.MX.sym('x',5);


y = ad.norm(x,2);

grad_y = ad.gradient(y,x)

f = ad.Function('f',{x},{grad_y});


grad_y_num = f([1,2,3,4,5])