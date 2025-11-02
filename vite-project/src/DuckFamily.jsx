import { motion } from "framer-motion";
import { useEffect, useState } from "react";

const Duck = ({ isMama }) => (
  <motion.div
    style={{
      width: isMama ? 50 : 30,
      height: isMama ? 50 : 30,
    }}
    animate={{ y: [0, -5, 0] }}
    transition={{ duration: 0.5, repeat: Infinity, repeatType: "reverse" }}
  >
    <img src={isMama ? "/Mama.png" : "/Baby.png"} alt={isMama ? "Mama Duck" : "Baby Duck"} style={{ width: "100%", height: "100%" }} />
  </motion.div>
);

const DuckFamily = ({ onDucksLeave }) => {
  const [key, setKey] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setKey((prevKey) => prevKey + 1);
    }, 20000);

    return () => clearInterval(interval);
  }, []);

  // Call onDucksLeave when animation completes (when ducks leave right edge)
  const handleAnimationComplete = () => {
    if (onDucksLeave) onDucksLeave();
  };

  return (
    <motion.div
      key={key}
      style={{ position: "absolute", top: "80%", display: "flex", alignItems: "center" }}
      initial={{ x: "-100%" }}
      animate={{ x: "100vw" }}
      transition={{ duration: 10, ease: "linear" }}
      onAnimationComplete={handleAnimationComplete}
    >
      <Duck />
      <div style={{ width: 10 }} />
      <Duck />
      <div style={{ width: 10 }} />
      <Duck />
      <div style={{ width: 20 }} />
      <Duck isMama />
    </motion.div>
  );
};

export default DuckFamily;