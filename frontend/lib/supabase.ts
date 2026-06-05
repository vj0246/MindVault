import { createClient, SupabaseClient } from '@supabase/supabase-js'

let _supabase: SupabaseClient | null = null

export function getSupabase() {
  if (!_supabase) {
    _supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
    )
  }
  return _supabase
}

export const supabase = {
  auth: {
    getSession: () => getSupabase().auth.getSession(),
    getUser: () => getSupabase().auth.getUser(),
    signUp: (creds: any) => getSupabase().auth.signUp(creds),
    signInWithPassword: (creds: any) => getSupabase().auth.signInWithPassword(creds),
    signOut: () => getSupabase().auth.signOut(),
  }
}

export async function getUser() {
  const { data: { user } } = await getSupabase().auth.getUser()
  return user
}

export async function getToken() {
  const { data: { session } } = await getSupabase().auth.getSession()
  return session?.access_token || null
}

export async function signOut() {
  await getSupabase().auth.signOut()
  window.location.href = '/login'
}