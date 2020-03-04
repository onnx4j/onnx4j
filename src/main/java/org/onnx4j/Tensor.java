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
package org.onnx4j;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.onnx4j.tensor.DataType;
import org.onnx4j.tensor.Shape;
import org.onnx4j.tensor.TensorDump;
import org.onnx4j.tensor.ValueInfo;
import org.onnx4j.utils.DirectBufferDealloc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Tensor extends NamedOnnxObject implements AutoCloseable {

	private static Logger logger = LoggerFactory.getLogger(Tensor.class);

	public enum AllocationMode {

		DIRECT, HEAP

	}

	public static class Options {

		private AllocationMode allocationMode = AllocationMode.DIRECT;

		private ByteOrder byteOrder = ByteOrder.nativeOrder();

		private Options() {
		}

		public AllocationMode getAllocationMode() {
			return allocationMode;
		}

		public Options setAllocationMode(AllocationMode allocationMode) {
			this.allocationMode = allocationMode;
			return this;
		}

		public ByteOrder getByteOrder() {
			return byteOrder;
		}

		public Options setByteOrder(ByteOrder byteOrder) {
			this.byteOrder = byteOrder;
			return this;
		}

	}

	public static Options options() {
		return new Options();
	}

	private ValueInfo valueInfo;
	private ByteBuffer dataBuffer;

	public Tensor(String name, String docString, DataType dataType, Shape shape, ByteBuffer dataBuffer) {
		super(name, docString);

		if (dataBuffer == null || dataBuffer.capacity() <= 0)
			throw new IllegalArgumentException("Databuffer is null or empty");

		this.name = name;
		this.valueInfo = new ValueInfo(dataType, shape);
		this.dataBuffer = dataBuffer;
	}

	public String getName() {
		return name;
	}

	/**
	 * 获取Tensor数据部分(ByteBuffer)的内存占用量
	 * 
	 * @return 占用字节数
	 */
	public long getMemoryBytes() {
		if (this.dataBuffer != null)
			return this.dataBuffer.capacity();

		return -1L;
	}

	public ValueInfo getValueInfo() {
		return valueInfo;
	}

	/**
	 * 获取元素总数量
	 * 
	 * @return 元素总数量
	 */
	public long getElementSize() {
		return this.getMemoryBytes() / this.valueInfo.getDataType().getUnitSize();
	}

	public DataType getDataType() {
		return this.valueInfo.getDataType();
	}

	public int getRanks() {
		return this.valueInfo.getRank();
	}

	/**
	 * 返回当前数据缓存的只读引用
	 * 
	 * @return
	 */
	public ByteBuffer getData() {
		return this.dataBuffer.slice().asReadOnlyBuffer().order(this.dataBuffer.order());
	}

	public long[] getShape() {
		return this.valueInfo.getShape().toArray();
	}

	public boolean equalsIn(DataType[] constrainTypes) {
		for (DataType dataType : constrainTypes) {
			if (dataType.equals(this.valueInfo.getDataType()))
				return true;
		}
		return false;
	}

	public boolean equals(ValueInfo valueInfo) {
		return this.valueInfo.equals(valueInfo);
	}

	@Override
	public boolean equals(Object obj) {
		if (obj.getClass().isInstance(Tensor.class) == false)
			return false;

		if (this.equals(((Tensor) obj).valueInfo) == false)
			return false;

		return this.dataBuffer.equals(((Tensor) obj).dataBuffer);
	}

	@Override
	public void close() {
		if (this.dataBuffer != null) {
			if (!DirectBufferDealloc.deallocateDirectBuffer(this.dataBuffer))
				throw new RuntimeException(String.format("[Tensor:%s] can not be released.", this.name));

			this.dataBuffer = null;
		}
	}

	@Override
	public String toString() {
		return TensorDump.dump(this);
	}

}