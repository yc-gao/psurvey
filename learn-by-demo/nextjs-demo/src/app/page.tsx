import { BeakerIcon } from '@heroicons/react/24/solid'
import { Button } from '@/components/ui/Button'

export default function Page() {
  return <div>
    <Button>button default</Button>
    <Button variant="primary">button primary</Button>
    <Button variant="secondary" size="lg">button secondary</Button>
    <BeakerIcon className="size-6 text-blue-500" />
  </div>
}
