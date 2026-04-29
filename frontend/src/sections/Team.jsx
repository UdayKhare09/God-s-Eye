import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Github, Linkedin, Mail } from "lucide-react";

/* ---------- reactive mobile detector ---------- */
const useIsMobile = () => {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia("(max-width: 767px)");
    const handler = () => setIsMobile(mq.matches);
    handler();
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  return isMobile;
};

const teamMembers = [
  { id: 1, name: "Tanishq", role: "Lead Developer", image: "/team/TanishqCharacterPassport.png", socials:{linkedin:"#",github:"#",discord:"#"} },
  { id: 2, name: "Uday", role: "System Arch.", image: "/team/TanishqCharacterPassport.png", socials:{linkedin:"#",github:"#",discord:"#"} },
  { id: 3, name: "Surendra", role: "Neural Ops", image: "/team/TanishqCharacterPassport.png", socials:{linkedin:"#",github:"#",discord:"#"} },
  { id: 4, name: "Vansh", role: "Security", image: "/team/TanishqCharacterPassport.png", socials:{linkedin:"#",github:"#",discord:"#"} },
];

const Team = () => {
  const [active, setActive] = useState(null);
  const isMobile = useIsMobile();

  return (
    <section id="team" className="py-14 md:py-20 relative bg-transparent text-neutral-900 overflow-visible">
      <div className="container mx-auto px-4 md:px-6 relative z-10">

        {/* Header */}
        <motion.div
          initial={{ opacity:0, y:20 }}
          whileInView={{ opacity:1, y:0 }}
          transition={{ duration:0.6 }}
          viewport={{ once:true }}
          className="mb-12 md:mb-20 text-center"
        >
          <h3 className="text-3xl md:text-5xl font-bold mb-4 tracking-tighter">
            Active Personnel
          </h3>
          <p className="text-neutral-600 max-w-lg mx-auto font-mono text-xs md:text-sm px-2 md:px-8">
            The core team behind the project, each with unique expertise and a shared vision for secure AI deployment.
          </p>
        </motion.div>

        {/* GRID */}
        <div className="
          grid
          grid-cols-2 md:grid-cols-2 lg:grid-cols-4
          gap-5 md:gap-x-8 md:gap-y-12
          justify-items-center
        ">
          {teamMembers.map((member, index) => {

            const isOpen = active === index;

            return (
              <div
                key={member.id}
                className="group relative w-full max-w-[165px] md:max-w-none md:w-64 md:h-80"
                onMouseEnter={() => !isMobile && setActive(index)}
                onMouseLeave={() => !isMobile && setActive(null)}
                onClick={() => isMobile && setActive(isOpen ? null : index)}
              >
                {/* Card */}
                <div className="
                  relative w-full
                  h-56 md:h-full
                  bg-[#e8e6df]
                  rounded-xl
                  transition-all duration-500
                  flex flex-col items-center
                  overflow-visible
                  border border-neutral-300
                ">

                  {/* background pattern */}
                  <div className="absolute inset-0 opacity-10 bg-[radial-gradient(#444_1px,transparent_1px)] bg-[length:14px_14px] rounded-xl pointer-events-none"></div>

                  {/* image */}
                  <div className="relative w-full flex items-end justify-center z-20 pt-4 md:pb-16 overflow-visible">
                    <img
                      src={member.image}
                      alt={member.name}
                      className="
                        relative
                        w-28 md:w-48
                        object-cover object-top
                        transition-all duration-500
                        origin-bottom
                        transform
                        md:group-hover:scale-115 md:group-hover:-translate-y-2
                        drop-shadow-md
                      "
                    />
                  </div>

                  {/* info panel */}
                  <div className="
                    absolute bottom-0 left-0 w-full
                    bg-white
                    h-16 md:h-20
                    rounded-b-xl
                    border-t border-neutral-200
                    flex flex-col items-center justify-center
                    transition-all duration-300
                    z-20
                  ">
                    <div className="text-center px-2 w-full">
                      <h4 className="text-sm md:text-lg font-bold leading-tight">
                        {member.name}
                      </h4>
                      <p className="text-[10px] md:text-xs text-neutral-500 font-mono tracking-wider uppercase">
                        {member.role}
                      </p>
                    </div>

                    {/* socials */}
                    <div className={`
                      absolute bottom-2 flex gap-3
                      transition-all duration-300
                      ${isOpen
                        ? "opacity-100 translate-y-0"
                        : "opacity-0 translate-y-2 md:group-hover:opacity-100 md:group-hover:translate-y-0"}
                    `}>
                      <a href={member.socials.linkedin} className="text-neutral-500 hover:text-black">
                        <Linkedin size={10}/>
                      </a>
                      <a href={member.socials.github} className="text-neutral-500 hover:text-black">
                        <Github size={10}/>
                      </a>
                      <a href={member.socials.discord} className="text-neutral-500 hover:text-black">
                        <Mail size={10}/>
                      </a>
                    </div>

                  </div>
                </div>
              </div>
            );
          })}
        </div>

      </div>
    </section>
  );
};

export default Team;