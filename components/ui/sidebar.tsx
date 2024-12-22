"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

const Sidebar = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn("flex flex-col h-screen w-64 bg-[#0f0a19] text-white", className)}
      {...props}
    />
  )
})
Sidebar.displayName = "Sidebar"

const SidebarSection = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn("px-3 py-2", className)}
      {...props}
    />
  )
})
SidebarSection.displayName = "SidebarSection"

const SidebarItem = React.forwardRef<
  HTMLAnchorElement,
  React.AnchorHTMLAttributes<HTMLAnchorElement> & { active?: boolean }
>(({ className, active, ...props }, ref) => {
  return (
    <a
      ref={ref}
      className={cn(
        "flex items-center px-3 py-2 text-sm rounded-md",
        active ? "bg-[#1c1528] text-white" : "text-gray-400 hover:bg-[#1c1528] hover:text-white",
        className
      )}
      {...props}
    />
  )
})
SidebarItem.displayName = "SidebarItem"

export { Sidebar, SidebarSection, SidebarItem }

