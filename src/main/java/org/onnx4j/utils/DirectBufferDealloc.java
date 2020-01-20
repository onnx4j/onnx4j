package org.onnx4j.utils;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.Buffer;
import java.util.Collection;

public class DirectBufferDealloc {

	protected static final Class<?> directBufferClass;
	protected static final Method getCleanerMethod;
	protected static final Method getAttachmentMethod;

	protected static final Class<?> cleanerClass;
	protected static final Method doCleanMethod;

	static {
		try {
			directBufferClass = Class.forName("sun.nio.ch.DirectBuffer");
			getCleanerMethod = directBufferClass.getMethod("cleaner");
			getAttachmentMethod = directBufferClass.getMethod("attachment");

			cleanerClass = Class.forName("sun.misc.Cleaner");
			doCleanMethod = cleanerClass.getMethod("clean");
		} catch (ClassNotFoundException | NoSuchMethodException e) {

			throw new RuntimeException(e);
		}
	}

	public static void deallocateDirectBuffers(Collection<? extends Buffer> directBuffers) {
		if (directBuffers != null) {
			for (Buffer b : directBuffers) {
				deallocateDirectBuffer0(b);
			}
		}
	}

	public static void deallocateDirectBuffers(Buffer... directBuffers) {
		if (directBuffers != null) {
			for (Buffer b : directBuffers) {
				deallocateDirectBuffer0(b);
			}
		}
	}

	public static boolean deallocateDirectBuffer(Buffer directBuffer) {
		return deallocateDirectBuffer0(directBuffer);
	}

	public static boolean deallocateDirectBuffer0(Object directBuffer) {
		if (directBufferClass.isInstance(directBuffer)) {
			try {
				Object cleaner = getCleanerMethod.invoke(directBuffer);
				if (cleanerClass.isInstance(cleaner)) {
					doCleanMethod.invoke(cleaner);
				} else {
					return false;
				}

				Object attachment = getAttachmentMethod.invoke(directBuffer);
				deallocateDirectBuffer0(attachment);
				return true;
			} catch (IllegalAccessException | IllegalArgumentException | InvocationTargetException e) {
				throw new RuntimeException(e);
			}
		} else {
			return false;
		}
	}

}
