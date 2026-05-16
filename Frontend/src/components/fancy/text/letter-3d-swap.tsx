"use client"

import React, { ElementType, useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  AnimationOptions,
  useAnimate,
  ValueAnimationTransition,
} from "motion/react"

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
  staggerDuration?: number
  staggerFrom?: "first" | "last" | "center" | number | "random"
  transition?: ValueAnimationTransition | AnimationOptions
  rotateDirection?: "top" | "right" | "bottom" | "left"
  auto?: boolean
  rotationInterval?: number
}

const Letter3DSwap = ({
  children,
  texts,
  as = "p",
  mainClassName,
  frontFaceClassName,
  secondFaceClassName,
  staggerDuration = 0.05,
  staggerFrom = "first",
  transition = { type: "spring", damping: 30, stiffness: 300 },
  rotateDirection = "top",
  auto = false,
  rotationInterval = 3000,
  ...props
}: Letter3DSwapProps) => {
  const [isAnimating, setIsAnimating] = useState(false)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [scope, animate] = useAnimate()
  const autoTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const rotationTransform = (() => {
    switch (rotateDirection) {
      case "top":
        return "rotateX(90deg)"
      case "right":
        return "rotateY(90deg)"
      case "bottom":
        return "rotateX(-90deg)"
      case "left":
        return "rotateY(-90deg)"
      default:
        return "rotateX(90deg)"
    }
  })()

  // Determine current text to display
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

  const getStaggerDelay = useCallback(
    (index: number, totalChars: number) => {
      const total = totalChars
      if (staggerFrom === "first") return index * staggerDuration
      if (staggerFrom === "last") return (total - 1 - index) * staggerDuration
      if (staggerFrom === "center") {
        const center = Math.floor(total / 2)
        return Math.abs(center - index) * staggerDuration
      }
      if (staggerFrom === "random") {
        const randomIndex = Math.floor(Math.random() * total)
        return Math.abs(randomIndex - index) * staggerDuration
      }
      return Math.abs(staggerFrom - index) * staggerDuration
    },
    [staggerFrom, staggerDuration]
  )

  const triggerAnimation = useCallback(async () => {
    if (isAnimating) return
    setIsAnimating(true)

    const totalChars = characters.reduce(
      (sum: number, word: WordObject) => sum + word.characters.length,
      0
    )

    const delays = Array.from({ length: totalChars }, (_, i) =>
      getStaggerDelay(i, totalChars)
    )

    await animate(
      ".letter-3d-swap-char-box-item",
      { transform: rotationTransform },
      {
        ...transition,
        delay: (i: number) => delays[i],
      }
    )

    // Advance to next text while rotated (hidden)
    if (texts && texts.length > 0) {
      setCurrentIndex((prev) => (prev + 1) % texts.length)
    }

    // Brief pause at peak rotation so text swap is invisible
    await new Promise((r) => setTimeout(r, 120))

    // Reset all boxes
    await animate(
      ".letter-3d-swap-char-box-item",
      { transform: "rotateX(0deg) rotateY(0deg)" },
      { duration: 0 }
    )

    setIsAnimating(false)
  }, [isAnimating, characters, transition, getStaggerDelay, rotationTransform, animate, texts])

  // Auto-cycling
  useEffect(() => {
    if (!auto || !texts || texts.length <= 1) return
    autoTimerRef.current = setInterval(() => {
      triggerAnimation()
    }, rotationInterval)
    return () => {
      if (autoTimerRef.current) clearInterval(autoTimerRef.current)
    }
  }, [auto, texts, rotationInterval, triggerAnimation])

  // Trigger initial animation on mount for auto mode
  useEffect(() => {
    if (auto && texts && texts.length > 0) {
      const timeout = setTimeout(() => triggerAnimation(), 600)
      return () => clearTimeout(timeout)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleHoverStart = useCallback(() => {
    if (auto) return // auto mode manages its own animation
    triggerAnimation()
  }, [auto, triggerAnimation])

  const ElementTag = as ?? "p"

  return (
    <ElementTag
      className={cn("flex flex-wrap relative", mainClassName)}
      onMouseEnter={handleHoverStart}
      ref={scope}
      {...props}
    >
      <span className="sr-only">{currentText}</span>

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
                    key={`${currentIndex}-${totalIndex}`}
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
      className="letter-3d-swap-char-box-item inline-block"
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
