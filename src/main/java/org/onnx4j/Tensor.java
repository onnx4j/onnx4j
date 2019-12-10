package org.onnx4j;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Consumer;

import org.onnx4j.onnx.prototypes.OnnxProto3.TensorProto;
import org.onnx4j.tensor.DataType;
import org.onnx4j.tensor.Shape;
import org.onnx4j.tensor.TensorDump;
import org.onnx4j.tensor.ValueInfo;
import org.onnx4j.utils.PrimitiveUtil;

public class Tensor implements AutoCloseable {

	private ValueInfo valueInfo;
	private ByteBuffer dataBuffer;

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

	public static class Builder {

		private DataType dataType;
		private Shape shape;
		private ByteBuffer dataBuffer;

		public Builder(ByteBuffer dataBuffer) {
			this.dataBuffer = dataBuffer;
		}

		public Builder(DataType dataType, Shape shape, Tensor.Options options) {
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

		public Builder write(Consumer<ByteBuffer> consumer) {
			consumer.accept(this.dataBuffer);
			return this;
		}

		public Builder put(byte[] src) {
			this.dataBuffer.put(src);
			return this;
		}

		public Builder putFloat(Float f) {
			this.dataBuffer.putFloat(f);
			return this;
		}

		public Tensor build() {
			this.dataBuffer.rewind();
			return new Tensor(this.dataType, this.shape, this.dataBuffer);
		}

	}

	public static Options options() {
		return new Options();
	}
	
	public static Builder builder(DataType dataType, Shape shape, ByteBuffer dataBuffer) {
		return new Builder(dataBuffer);
	}
	
	public static Builder builder(DataType dataType, Shape shape, Tensor.Options options) {
		return new Builder(dataType, shape, options);
	}

	public static List<Tensor> toTensors(List<TensorProto> tensorProtos) {
		return Tensor.toTensors(tensorProtos, Tensor.options());
	}

	public static List<Tensor> toTensors(List<TensorProto> tensorProtos, Tensor.Options options) {
		List<Tensor> tensors = new LinkedList<Tensor>();
		for (TensorProto tensorProto : tensorProtos) {
			tensors.add(Tensor.toTensor(tensorProto));
		}
		return tensors;
	}

	public static Tensor toTensor(TensorProto tensorProto) {
		return Tensor.toTensor(tensorProto, Tensor.options());
	}

	public static Tensor toTensor(TensorProto tensorProto, Tensor.Options options) {
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
		
		return Tensor.builder(dataType, shape, options).put(byteArray).build();
	}

	public Tensor(DataType dataType, Shape shape, ByteBuffer dataBuffer) {
		if (dataBuffer == null || dataBuffer.capacity() <= 0)
			throw new IllegalArgumentException("Data buffer is null or empty");

		this.valueInfo = new ValueInfo(dataType, shape);
		this.dataBuffer = dataBuffer;
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
			this.dataBuffer = null;
		}
	}

	@Override
	public String toString() {
		return TensorDump.dump(this);
	}

}
