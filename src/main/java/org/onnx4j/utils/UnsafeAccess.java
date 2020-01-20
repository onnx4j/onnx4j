package org.onnx4j.utils;

import java.lang.reflect.Field;
import java.security.AccessController;
import java.security.PrivilegedExceptionAction;

/**
 * The idea of allocating the unsafe statically and 
 * checking access privilige is borrowed from the Agrona project.
 *
 */
import sun.misc.Unsafe;

// TODO: Auto-generated Javadoc
/**
 * The Class UnsafeAccess.
 */
@SuppressWarnings("restriction")
public class UnsafeAccess {

	/** The unsafe. */
	public static Unsafe UNSAFE;

	/**
	 * This method is guaranteed to create unsafe reference statically when the
	 * class loader is being called.
	 * 
	 * It does not need to be synchronized because class loader is synchronizing
	 * this action.
	 */

	static {
		Unsafe unsafe = null;
		try {
			final PrivilegedExceptionAction<Unsafe> action = () -> {
				final Field f = Unsafe.class.getDeclaredField("theUnsafe");
				f.setAccessible(true);
				return (Unsafe) f.get(null);
			};

			unsafe = AccessController.doPrivileged(action);
		} catch (final Exception ex) {
			throw new RuntimeException(ex);
		}

		UNSAFE = unsafe;
	}

	/**
	 * Create and return the unsafe.
	 *
	 * @return the unsafe
	 * @throws MemoryManagerException
	 *             the memory manager exception
	 */
	public static sun.misc.Unsafe getUnsafe() throws RuntimeException {
		try {
			if (UNSAFE == null) {
				// to speed this call up
				synchronized (UnsafeAccess.class) {
					// check again to prevent racing
					if (UNSAFE == null) {
						Field f = sun.misc.Unsafe.class.getDeclaredField("theUnsafe");
						f.setAccessible(true);
						UNSAFE = (sun.misc.Unsafe) f.get(null);
					}
				}
			}
			return UNSAFE;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
}
