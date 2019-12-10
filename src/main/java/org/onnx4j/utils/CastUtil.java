package org.onnx4j.utils;

public class CastUtil {

	public static <T> T cast(Object object, Class<T> typeOfTarget) {
		if (object.getClass().isInstance(typeOfTarget)) {
			return typeOfTarget.cast(object);
		} else {
			throw new ClassCastException(String.format(
					"Object is not a instance of %s.", typeOfTarget.getClass()
							.getName()));
		}
	}
}
