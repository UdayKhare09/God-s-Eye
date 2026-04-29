import{a as e}from"./rolldown-runtime-Cn8xt2Gj.js";import{d as t,l as n}from"./drei-P_yk8uxv.js";import{r}from"./framer-motion-CumvveF2.js";import{n as i}from"./index-BtJknGRJ.js";var a=i(`github`,[[`path`,{d:`M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4`,key:`tonef`}],[`path`,{d:`M9 18c-4.51 2-5-2-7-2`,key:`9comsn`}]]),o=i(`linkedin`,[[`path`,{d:`M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z`,key:`c2jq9f`}],[`rect`,{width:`4`,height:`12`,x:`2`,y:`9`,key:`mk3on5`}],[`circle`,{cx:`4`,cy:`4`,r:`2`,key:`bt5ra8`}]]),s=i(`mail`,[[`path`,{d:`m22 7-8.991 5.727a2 2 0 0 1-2.009 0L2 7`,key:`132q7q`}],[`rect`,{x:`2`,y:`4`,width:`20`,height:`16`,rx:`2`,key:`izxlao`}]]),c=e(t(),1),l=n(),u=()=>{let[e,t]=(0,c.useState)(!1);return(0,c.useEffect)(()=>{let e=window.matchMedia(`(max-width: 767px)`),n=()=>t(e.matches);return n(),e.addEventListener(`change`,n),()=>e.removeEventListener(`change`,n)},[]),e},d=[{id:1,name:`Tanishq`,role:`Lead Developer`,image:`/team/TanishqCharacterPassport.png`,socials:{linkedin:`#`,github:`#`,discord:`#`}},{id:2,name:`Uday`,role:`System Arch.`,image:`/team/TanishqCharacterPassport.png`,socials:{linkedin:`#`,github:`#`,discord:`#`}},{id:3,name:`Surendra`,role:`Neural Ops`,image:`/team/TanishqCharacterPassport.png`,socials:{linkedin:`#`,github:`#`,discord:`#`}},{id:4,name:`Vansh`,role:`Security`,image:`/team/TanishqCharacterPassport.png`,socials:{linkedin:`#`,github:`#`,discord:`#`}}],f=()=>{let[e,t]=(0,c.useState)(null),n=u();return(0,l.jsx)(`section`,{id:`team`,className:`py-14 md:py-20 relative bg-transparent text-neutral-900 overflow-visible`,children:(0,l.jsxs)(`div`,{className:`container mx-auto px-4 md:px-6 relative z-10`,children:[(0,l.jsxs)(r.div,{initial:{opacity:0,y:20},whileInView:{opacity:1,y:0},transition:{duration:.6},viewport:{once:!0},className:`mb-12 md:mb-20 text-center`,children:[(0,l.jsx)(`h3`,{className:`text-3xl md:text-5xl font-bold mb-4 tracking-tighter`,children:`Active Personnel`}),(0,l.jsx)(`p`,{className:`text-neutral-600 max-w-lg mx-auto font-mono text-xs md:text-sm px-2 md:px-8`,children:`The core team behind the project, each with unique expertise and a shared vision for secure AI deployment.`})]}),(0,l.jsx)(`div`,{className:`
          grid
          grid-cols-2 md:grid-cols-2 lg:grid-cols-4
          gap-5 md:gap-x-8 md:gap-y-12
          justify-items-center
        `,children:d.map((r,i)=>{let c=e===i;return(0,l.jsx)(`div`,{className:`group relative w-full max-w-[165px] md:max-w-none md:w-64 md:h-80`,onMouseEnter:()=>!n&&t(i),onMouseLeave:()=>!n&&t(null),onClick:()=>n&&t(c?null:i),children:(0,l.jsxs)(`div`,{className:`
                  relative w-full
                  h-56 md:h-full
                  bg-[#e8e6df]
                  rounded-xl
                  transition-all duration-500
                  flex flex-col items-center
                  overflow-visible
                  border border-neutral-300
                `,children:[(0,l.jsx)(`div`,{className:`absolute inset-0 opacity-10 bg-[radial-gradient(#444_1px,transparent_1px)] bg-[length:14px_14px] rounded-xl pointer-events-none`}),(0,l.jsx)(`div`,{className:`relative w-full flex items-end justify-center z-20 pt-4 md:pb-16 overflow-visible`,children:(0,l.jsx)(`img`,{src:r.image,alt:r.name,className:`
                        relative
                        w-28 md:w-48
                        object-cover object-top
                        transition-all duration-500
                        origin-bottom
                        transform
                        md:group-hover:scale-115 md:group-hover:-translate-y-2
                        drop-shadow-md
                      `})}),(0,l.jsxs)(`div`,{className:`
                    absolute bottom-0 left-0 w-full
                    bg-white
                    h-16 md:h-20
                    rounded-b-xl
                    border-t border-neutral-200
                    flex flex-col items-center justify-center
                    transition-all duration-300
                    z-20
                  `,children:[(0,l.jsxs)(`div`,{className:`text-center px-2 w-full`,children:[(0,l.jsx)(`h4`,{className:`text-sm md:text-lg font-bold leading-tight`,children:r.name}),(0,l.jsx)(`p`,{className:`text-[10px] md:text-xs text-neutral-500 font-mono tracking-wider uppercase`,children:r.role})]}),(0,l.jsxs)(`div`,{className:`
                      absolute bottom-2 flex gap-3
                      transition-all duration-300
                      ${c?`opacity-100 translate-y-0`:`opacity-0 translate-y-2 md:group-hover:opacity-100 md:group-hover:translate-y-0`}
                    `,children:[(0,l.jsx)(`a`,{href:r.socials.linkedin,className:`text-neutral-500 hover:text-black`,children:(0,l.jsx)(o,{size:10})}),(0,l.jsx)(`a`,{href:r.socials.github,className:`text-neutral-500 hover:text-black`,children:(0,l.jsx)(a,{size:10})}),(0,l.jsx)(`a`,{href:r.socials.discord,className:`text-neutral-500 hover:text-black`,children:(0,l.jsx)(s,{size:10})})]})]})]})},r.id)})})]})})};export{f as default};