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

import java.nio.ByteBuffer;
import java.util.function.Consumer;

import org.onnx4j.Tensor;
import org.onnx4j.Tensor.AllocationMode;
import org.onnx4j.TensorManager;
import org.onnx4j.prototypes.OnnxProto3.TensorProto;
import org.onnx4j.utils.PrimitiveUtil;

public class TensorBuilder {

	private TensorManager<Tensor> tensorManager;
	private String name;
	private String docString;
	private DataType dataType;
	private Shape shape;
	private ByteBuffer dataBuffer;

	public TensorBuilder(DataType dataType, Shape shape, ByteBuffer dataBuffer) {
		this.dataBuffer = dataBuffer;
		this.dataType = dataType;
		this.shape = shape;
	}

	public TensorBuilder(DataType dataType, Shape shape, Tensor.Options options) {
		int size = (int) shape.numElements() * dataType.getUnitSize();

		if (size <= 0)
			throw new IllegalArgumentException(String.format("Can not to allocate memory with size=%s", size));

		if (options == null)
			throw new IllegalArgumentException(
					"Argument named \"options\" is null,try \"Tensor.options()\" for default instead.");

		ByteBuffer byteBuffer = null;
		if (AllocationMode.DIRECT == options.getAllocationMode())
			byteBuffer = ByteBuffer.allocateDirect(size);
		else if (AllocationMode.HEAP == options.getAllocationMode())
			byteBuffer = ByteBuffer.allocate(size);
		else
			throw new UnsupportedOperationException(
					String.format("Unsupported memory allocation mode for \"%s\"", options.getAllocationMode()));

		this.dataBuffer = byteBuffer.order(options.getByteOrder());
		this.dataType = dataType;
		this.shape = shape;
	}

	public TensorBuilder manager(TensorManager<Tensor> tensorManager) {
		this.tensorManager = tensorManager;
		return this;
	}

	public TensorBuilder name(String name) {
		this.name = name;
		return this;
	}

	public TensorBuilder docString(String docString) {
		this.docString = docString;
		return this;
	}

	public TensorBuilder write(Consumer<ByteBuffer> consumer) {
		consumer.accept(this.dataBuffer);
		return this;
	}

	public TensorBuilder put(byte[] src) {
		this.dataBuffer.put(src);
		return this;
	}

	public TensorBuilder putFloat(Float f) {
		this.dataBuffer.putFloat(f);
		return this;
	}

	public Tensor build() {
		this.dataBuffer.rewind();
		Tensor tensor = new Tensor(this.name, this.docString, this.dataType, this.shape, this.dataBuffer);

		if (this.tensorManager != null)
			this.tensorManager.attach(this.name, tensor);

		return tensor;
	}

	public static TensorBuilder builder(DataType dataType, Shape shape, ByteBuffer dataBuffer) {
		return new TensorBuilder(dataType, shape, dataBuffer);
	}

	public static TensorBuilder builder(DataType dataType, Shape shape, Tensor.Options options) {
		return new TensorBuilder(dataType, shape, options);
	}

	public static TensorBuilder builder(TensorProto tensorProto) {
		return TensorBuilder.builder(tensorProto, Tensor.options());
	}

	public static TensorBuilder builder(TensorProto tensorProto, Tensor.Options options) {
		byte[] byteArray;
		Shape shape;
		DataType dataType;

		if (TensorProto.DataType.FLOAT.getNumber() == tensorProto.getDataType()) {
			byteArray = (tensorProto.getRawData() != null && tensorProto.getRawData().size() > 0)
					? tensorProto.getRawData().toByteArray()
					: PrimitiveUtil.toByteArrayFromFloats(tensorProto.getFloatDataList());
			shape = (tensorProto.getDimsCount() > 0) ? Shape.create(tensorProto.getDimsList())
					: Shape.create(new Long(byteArray.length / DataType.FLOAT.getUnitSize()));
			dataType = DataType.FLOAT;
		} else if (TensorProto.DataType.INT32.getNumber() == tensorProto.getDataType()) {
			byteArray = (tensorProto.getRawData() != null && tensorProto.getRawData().size() > 0)
					? tensorProto.getRawData().toByteArray()
					: PrimitiveUtil.toListToByteArrayFromInts(tensorProto.getInt32DataList());
			shape = (tensorProto.getDimsCount() > 0) ? Shape.create(tensorProto.getDimsList())
					: Shape.create(new Long(byteArray.length / DataType.INT32.getUnitSize()));
			dataType = DataType.INT32;
		} else if (TensorProto.DataType.INT64.getNumber() == tensorProto.getDataType()) {
			byteArray = (tensorProto.getRawData() != null && tensorProto.getRawData().size() > 0)
					? tensorProto.getRawData().toByteArray()
					: PrimitiveUtil.toListToByteArrayFromLongs(tensorProto.getInt64DataList());
			shape = (tensorProto.getDimsCount() > 0) ? Shape.create(tensorProto.getDimsList())
					: Shape.create(new Long(byteArray.length / DataType.INT64.getUnitSize()));
			dataType = DataType.INT64;
		} else {
			throw new UnsupportedOperationException(
					"Unsupported to handle data type: " + TensorProto.DataType.forNumber(tensorProto.getDataType()));
		}

		return TensorBuilder
				.builder(dataType, shape, options)
				.name(tensorProto.getName())
				.docString(tensorProto.getDocString())
				.put(byteArray);
	}
}
