import { motion, useAnimation } from "framer-motion";
import { useState, useEffect, useRef } from "react";

const BouncingShape = ({ constraintsRef }) => {
  const [isBouncing, setIsBouncing] = useState(false);
  const controls = useAnimation();
  const timeoutRef = useRef(null);

  const resetTimer = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    timeoutRef.current = setTimeout(() => {
      controls.start({
        x: 0,
        y: 0,
        transition: { type: "tween", duration: 1, ease: "easeInOut" },
      });
    }, 2000);
  };

  useEffect(() => {
    resetTimer();
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (isBouncing) {
      controls.start({
        rotate: 360,
        transition: { duration: 2, repeat: Infinity, ease: "linear" },
      });
    } else {
      controls.start({
        rotate: 0,
        transition: { duration: 0.5 },
      });
    }
  }, [isBouncing, controls]);

  const handleClick = () => {
    setIsBouncing(!isBouncing);
    resetTimer();
  };

  const handleDragStart = () => {
    resetTimer();
  };

  return (
    <motion.div
      className="absolute cursor-pointer"
      style={{
        width: 100,
        height: 100,
        background: "#ff0055",
        borderRadius: "50%",
        top: "calc(50% - 50px)",
        left: "calc(50% - 50px)",
      }}
      onClick={handleClick}
      drag
      dragConstraints={constraintsRef}
      dragTransition={{ bounceStiffness: 600, bounceDamping: 10 }}
      onDragStart={handleDragStart}
      animate={controls}
    />
  );
};

export default BouncingShape;