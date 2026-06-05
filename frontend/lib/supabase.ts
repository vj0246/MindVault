import { createClient } from '@supabase/supabase-js'

export const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
)

export async function getUser() {
  const { data: { user } } = await supabase.auth.getUser()
  return user
}

export async function getToken() {
  const { data: { session } } = await supabase.auth.getSession()
  return session?.access_token || null
}

export async function signOut() {
  await supabase.auth.signOut()
  window.location.href = '/login'
}