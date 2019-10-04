use rand;

pub fn fill_random<T>(arr: &mut [T])
where
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    for a in arr.iter_mut() {
        *a = rand::random::<T>();
    }

}
