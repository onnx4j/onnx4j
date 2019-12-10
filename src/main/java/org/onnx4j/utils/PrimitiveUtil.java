package org.onnx4j.utils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

public class PrimitiveUtil {

	public static byte[] toByteArray(float[] floatArray) {
		ByteBuffer buffer = ByteBuffer.allocate(floatArray.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
		for (float f : floatArray) {
			buffer.putFloat(f);
		}
		return buffer.array();
	}

	public static byte[] toByteArrayFromFloats(List<Float> floatList) {
		ByteBuffer buffer = ByteBuffer.allocate(floatList.size() * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
		for (float f : floatList) {
			buffer.putFloat(f);
		}
		return buffer.array();
	}

	public static byte[] toListToByteArrayFromLongs(List<Long> longList) {
		ByteBuffer buffer = ByteBuffer.allocate(longList.size() * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
		for (long l : longList) {
			buffer.putLong(l);
		}
		return buffer.array();
	}

}
