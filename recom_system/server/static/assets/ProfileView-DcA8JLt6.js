import{i as t,r as u,j as i,h as c,o as f,f as p,k as _,u as m,_ as g,G as k,l as R}from"./index-CsKy0wzl.js";const B={__name:"ProfileView",setup(d){const a=t("user"),e=t("userRatings"),s=u({});i(async()=>{a.value=JSON.parse(localStorage.getItem("user")),e.value=await c(a.value.id)});function r(o){s.value=o;const{open:n,close:l}=k({component:R,attrs:{book:s.value,userRatings:e.value,onConfirm(){l()}}});n()}return(o,n)=>(f(),p("main",null,[_(g,{name:"Rated Books",type:"profile",userRatings:m(e),onDisplayBook:r,large:!1},null,8,["userRatings"])]))}};export{B as default};