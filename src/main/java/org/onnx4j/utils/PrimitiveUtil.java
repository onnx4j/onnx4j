/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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