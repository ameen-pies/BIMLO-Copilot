"use client"

import React, { ElementType, useCallback, useEffect, useMemo, useRef, useState } from "react"

import { cn } from "@/lib/utils"

const splitIntoCharacters = (text: string): string[] => {
  if (typeof Intl !== "undefined" && "Segmenter" in Intl) {
    const segmenter = new Intl.Segmenter("en", { granularity: "grapheme" })
    return Array.from(segmenter.segment(text), ({ segment }) => segment)
  }
  return Array.from(text)
}

const extractTextFromChildren = (children: React.ReactNode): string | undefined => {
  if (children == null) return ""
  if (typeof children === "string") return children
  if (typeof children === "number") return String(children)
  if (Array.isArray(children)) {
    return children.map(extractTextFromChildren).join("")
  }
  if (React.isValidElement(children)) {
    const props = (children as React.ReactElement).props
    const childText = (props as any).children as React.ReactNode
    if (childText != null) {
      return extractTextFromChildren(childText)
    }
    return ""
  }
}

interface WordObject {
  characters: string[]
  needsSpace: boolean
}

interface Letter3DSwapProps {
  children?: React.ReactNode
  texts?: string[]
  as?: ElementType
  mainClassName?: string
  frontFaceClassName?: string
  secondFaceClassName?: string
  rotateDirection?: "top" | "right" | "bottom" | "left"
  auto?: boolean
  rotationInterval?: number
  /** Duration of the flip-up/flip-down transition in ms */
  flipDuration?: number
}

const Letter3DSwap = ({
  children,
  texts,
  as = "p",
  mainClassName,
  frontFaceClassName,
  secondFaceClassName,
  rotateDirection = "top",
  auto = false,
  rotationInterval = 3000,
  flipDuration = 500,
  ...props
}: Letter3DSwapProps) => {
  const [isAnimating, setIsAnimating] = useState(false)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [rotateAngle, setRotateAngle] = useState(0)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const rotationAxis = rotateDirection === "top" || rotateDirection === "bottom" ? "X" : "Y"
  const rotationSign = rotateDirection === "bottom" || rotateDirection === "left" ? "-" : ""

  const currentText = useMemo(() => {
    if (texts && texts.length > 0) {
      return texts[currentIndex % texts.length]
    }
    try {
      return extractTextFromChildren(children) ?? ""
    } catch {
      return ""
    }
  }, [texts, currentIndex, children])

  const characters = useMemo(() => {
    const t = currentText?.split(" ") ?? []
    return t.map((word: string, i: number) => ({
      characters: splitIntoCharacters(word),
      needsSpace: i !== t.length - 1,
    }))
  }, [currentText])

  const triggerAnimation = useCallback(() => {
    if (isAnimating) return
    setIsAnimating(true)

    // Phase 1: flip up
    setRotateAngle(90)

    // Phase 2: swap text while edge-on (invisible)
    timerRef.current = setTimeout(() => {
      if (texts && texts.length > 0) {
        setCurrentIndex((prev) => (prev + 1) % texts.length)
      }
      // Phase 3: flip back to reveal new text
      setRotateAngle(0)
    }, flipDuration + 30)

    // Phase 4: done
    timerRef.current = setTimeout(() => {
      setIsAnimating(false)
    }, (flipDuration + 30) * 2)
  }, [isAnimating, texts, flipDuration])

  // Auto-cycling
  useEffect(() => {
    if (!auto || !texts || texts.length <= 1) return
    // Initial animation after mount
    const initTimeout = setTimeout(() => triggerAnimation(), 600)
    intervalRef.current = setInterval(() => {
      triggerAnimation()
    }, rotationInterval)
    return () => {
      clearTimeout(initTimeout)
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [auto, texts, rotationInterval, triggerAnimation])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [])

  const handleHoverStart = useCallback(() => {
    if (auto) return
    triggerAnimation()
  }, [auto, triggerAnimation])

  const ElementTag = as ?? "p"

  return (
    <ElementTag
      className={cn("flex flex-wrap relative", mainClassName)}
      onMouseEnter={handleHoverStart}
      style={{
        perspective: "600px",
      }}
      {...props}
    >
      <span className="sr-only">{currentText}</span>
      <span
        style={{
          display: "inline-flex",
          flexWrap: "wrap",
          transform: `rotate${rotationAxis}(${rotationSign}${rotateAngle}deg)`,
          transformStyle: "preserve-3d",
          transition: `transform ${flipDuration}ms cubic-bezier(0.4, 0, 0.2, 1)`,
        }}
      >
        {characters.map(
          (wordObj: WordObject, wordIndex: number, array: WordObject[]) => {
            const previousCharsCount = array
              .slice(0, wordIndex)
              .reduce(
                (sum: number, word: WordObject) => sum + word.characters.length,
                0
              )

            return (
              <span key={wordIndex} className="inline-flex">
                {wordObj.characters.map((char: string, charIndex: number) => {
                  const totalIndex = previousCharsCount + charIndex
                  return (
                    <CharBox
                      key={totalIndex}
                      char={char}
                      frontFaceClassName={frontFaceClassName}
                      secondFaceClassName={secondFaceClassName}
                      rotateDirection={rotateDirection}
                    />
                  )
                })}
                {wordObj.needsSpace && <span className="whitespace-pre"> </span>}
              </span>
            )
          }
        )}
      </span>
    </ElementTag>
  )
}

interface CharBoxProps {
  char: string
  frontFaceClassName?: string
  secondFaceClassName?: string
  rotateDirection: "top" | "right" | "bottom" | "left"
}

const CharBox = ({
  char,
  frontFaceClassName,
  secondFaceClassName,
  rotateDirection,
}: CharBoxProps) => {
  const getSecondFaceTransform = () => {
    switch (rotateDirection) {
      case "top":
        return "rotateX(-90deg) translateZ(0.5lh)"
      case "right":
        return "rotateY(90deg) translateX(50%) rotateY(-90deg) translateX(-50%) rotateY(-90deg) translateX(50%)"
      case "bottom":
        return "rotateX(90deg) translateZ(0.5lh)"
      case "left":
        return "rotateY(90deg) translateX(50%) rotateY(-90deg) translateX(50%) rotateY(-90deg) translateX(50%)"
      default:
        return "rotateX(-90deg) translateZ(0.5lh)"
    }
  }

  const containerTransform =
    rotateDirection === "top" || rotateDirection === "bottom"
      ? "translateZ(-0.5lh)"
      : "rotateY(90deg) translateX(50%) rotateY(-90deg)"

  const frontFaceTransform =
    rotateDirection === "top" || rotateDirection === "bottom"
      ? "translateZ(0.5lh)"
      : rotateDirection === "left"
        ? "rotateY(90deg) translateX(50%) rotateY(-90deg)"
        : "rotateY(-90deg) translateX(50%) rotateY(90deg)"

  return (
    <span
      className="inline-block"
      style={{
        transform: containerTransform,
        transformStyle: "preserve-3d",
      }}
    >
      <span
        className={cn("relative h-[1em]", frontFaceClassName)}
        style={{
          backfaceVisibility: "hidden",
          transform: frontFaceTransform,
        }}
      >
        {char}
      </span>
      <span
        className={cn(
          "absolute h-[1em] top-0 left-0",
          secondFaceClassName
        )}
        style={{
          backfaceVisibility: "hidden",
          transform: getSecondFaceTransform(),
        }}
      >
        {char}
      </span>
    </span>
  )
}

Letter3DSwap.displayName = "Letter3DSwap"

export default Letter3DSwap
