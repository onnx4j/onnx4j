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
package org.onnx4j.tensor;

import org.onnx4j.prototypes.OnnxProto3.TensorProto;

public enum DataType {
	
	UINT8(0, Short.class, Byte.BYTES),
	UINT16(1, Short.class, Short.BYTES),
	UINT32(2, Integer.class, Integer.BYTES),
	UINT64(3, Long.class, Long.BYTES),
	INT8(4, Short.class, Byte.BYTES),
	INT16(5, Short.class, Short.BYTES),
	INT32(6, Integer.class, Integer.BYTES),
	INT64(7, Long.class, Long.BYTES),
	FLOAT16(8, Float.class, Float.BYTES),
	FLOAT(9, Float.class, Float.BYTES),
	DOUBLE(10, Double.class, Double.BYTES),
	STRING(11, String.class, 3), // 3 bytes per UTF-8 String
	BOOL(12, Byte.class, Byte.BYTES),
	
	// NOT IMPLEMENT
	COMPLEX64(13, Void.class, -1),
	COMPLEX128(14, Void.class, -1);
	
	private int code;
	private int unitSize;
	private Class<?> protoType;
	
	public int getCode() {
		return this.code;
	}
	
	public Class<?> getPrototype() {
		return this.protoType;
	}
	
	public int getUnitSize() {
		return this.unitSize;
	}
	
	DataType(int code, Class<?> protoType, int typeSize) {
		this.code = code;
		this.protoType = protoType;
		this.unitSize = typeSize;
	}
	
	public static DataType from(TensorProto.DataType protoDataType) {
		switch (protoDataType.getNumber()) {
			case TensorProto.DataType.UINT8_VALUE: return DataType.UINT8;
			case TensorProto.DataType.UINT16_VALUE: return DataType.UINT16;
			case TensorProto.DataType.UINT32_VALUE: return DataType.UINT32;
			case TensorProto.DataType.UINT64_VALUE: return DataType.UINT64;
			case TensorProto.DataType.INT8_VALUE: return DataType.INT8;
			case TensorProto.DataType.INT16_VALUE: return DataType.INT16;
			case TensorProto.DataType.INT32_VALUE: return DataType.INT32;
			case TensorProto.DataType.INT64_VALUE: return DataType.INT64;
			case TensorProto.DataType.FLOAT16_VALUE: return DataType.FLOAT16;
			case TensorProto.DataType.FLOAT_VALUE: return DataType.FLOAT;
			case TensorProto.DataType.DOUBLE_VALUE: return DataType.DOUBLE;
			case TensorProto.DataType.STRING_VALUE: return DataType.STRING;
			case TensorProto.DataType.BOOL_VALUE: return DataType.BOOL;
			default: return null;
		}
	}
	
	public static DataType from(String nameOfProto) {
		TensorProto.DataType protoDataType = TensorProto.DataType.valueOf(
				TensorProto.DataType.class, nameOfProto);
		return DataType.from(protoDataType);
	}
	
	public static DataType from(int numOfProto) {
		TensorProto.DataType protoDataType = TensorProto.DataType.forNumber(numOfProto);
		return DataType.from(protoDataType);
	}
	
	/**
	 * any types
	 * @return
	 */
	public static DataType[] allTypes() {
		final DataType[] dataTypes = {
				DataType.UINT8,
				DataType.UINT16,
				DataType.UINT32,
				DataType.UINT64,
				DataType.INT8,
				DataType.INT16,
				DataType.INT32,
				DataType.INT64,
				DataType.FLOAT16,
				DataType.FLOAT, 
				DataType.DOUBLE, 
				DataType.STRING, 
				DataType.BOOL, 
				DataType.COMPLEX64, 
				DataType.COMPLEX128
			};
		return dataTypes;
	}
	
	/**
	 * types to float tensors
	 * 
	 * @return
	 */
	public static DataType[] floatTypes() {
		final DataType[] dataTypes = {
				DataType.FLOAT16,
				DataType.FLOAT, 
				DataType.DOUBLE
			};
		return dataTypes;
	}
	
	/**
	 * types to all numeric tensors
	 * 
	 * @return
	 */
	public static DataType[] numericTypes() {
		final DataType[] dataTypes = {
				DataType.UINT8,
				DataType.UINT16,
				DataType.UINT32,
				DataType.UINT64,
				DataType.INT8,
				DataType.INT16,
				DataType.INT32,
				DataType.INT64,
				DataType.FLOAT16,
				DataType.FLOAT, 
				DataType.DOUBLE
			};
		return dataTypes;
	}
	
	/**
	 * any types without string and complex
	 * @return
	 */
	public static DataType[] allTypesWithoutStringAndComplex() {
		final DataType[] dataTypes = {
				DataType.UINT8,
				DataType.UINT16,
				DataType.UINT32,
				DataType.UINT64,
				DataType.INT8,
				DataType.INT16,
				DataType.INT32,
				DataType.INT64,
				DataType.FLOAT16,
				DataType.FLOAT, 
				DataType.DOUBLE, 
				DataType.BOOL
			};
		return dataTypes;
	}

}