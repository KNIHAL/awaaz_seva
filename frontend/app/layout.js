import './globals.css'

export const metadata = {
  title: 'Awaaz Seva - Voice AI Assistant',
  description: 'Voice-first AI assistant for Hindi and English',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#9ECAE1] antialiased">
        {children}
      </body>
    </html>
  )
}
