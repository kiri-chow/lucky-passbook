import{i as n,r as u,e as c,g as l,o as _,c as m,f,h as p,_ as g,G as k,j as d}from"./index-Dwtj5tBk.js";const R={__name:"ProfileView",setup(v){const s=n("user"),e=n("userRatings"),a=u({});c(async()=>{s.value=JSON.parse(localStorage.getItem("user")),e.value=await l(s.value.id)});function r(o){a.value=o;const{open:t,close:i}=k({component:d,attrs:{title:"test",book:a.value,userRatings:e.value,onConfirm(){i()}}});t()}return(o,t)=>(_(),m("main",null,[f(g,{name:"Liked Books",userRatings:p(e),onDisplayBook:r},null,8,["userRatings"])]))}};export{R as default};
