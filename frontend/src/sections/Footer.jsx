import React from "react";

const Footer = () => {
  return (
    <footer className="w-full py-3 border-t border-neutral-900/10 bg-black text-center relative z-10 ">
      {/* Scoped CSS just for this component */}
      <style>
        {`
          .footer-signature::selection {
            color: white;
            background: rgba(255,255,255,0.18);
          }

             .zemo-signature::selection {
            color: white;
            background: rgba(255,255,255,0.18);
          }
        `}
      </style>

      <p className="text-neutral-500 font-mono text-sm zemo-signature">
        © Team Zemo. Made by{" "}
        <a
          href="https://github.com/tanishqtiwari7"
          target="_blank"
          rel="noopener noreferrer"
          className=" footer-signature font-signature text-2xl text-neutral-300 hover:text-white transition-colors duration-300"
        >
          Tanishq
        </a>
      </p>
    </footer>
  );
};

export default Footer;
